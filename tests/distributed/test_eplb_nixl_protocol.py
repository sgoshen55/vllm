# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL EPLB sync protocol tests; run without GPU or NIXL."""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import regex as re
import torch

import vllm.distributed.eplb.eplb_communicator as eplb_communicator
from vllm.distributed.eplb.eplb_communicator import (
    NixlEplbCommunicator,
    NixlEplbNotification,
    NixlEplbTransferTracker,
    TransferKey,
    create_eplb_communicator,
)
from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator

READY = NixlEplbNotification.READY
READ_DONE = NixlEplbNotification.READ_DONE

WORLD_SIZE = 2
LAYER = 3
EXPERT = 11
# Rank 0 holds expert 11, rank 1 holds expert 12; one local expert each.
OLD_INDICES = np.array([EXPERT, 12])
TENSOR = torch.zeros(4, dtype=torch.float32)


class FakeClock:
    def __init__(self, step: float = 0.0) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        now = self.now
        self.now += self.step
        return now


class FakeNixlAgent:
    def __init__(self, names_as_bytes: bool = False) -> None:
        self.names_as_bytes = names_as_bytes
        self.released_xfers: list[int] = []
        self.released_dlists: list[int] = []
        self.sent_notifications: list[tuple[str | bytes, bytes]] = []
        self.notifications: dict[str | bytes, list[bytes]] = {}
        self.poll_calls = 0
        self.transfer_calls: list[int] = []
        self.check_calls: list[int] = []
        self.prepped_xfers: list[dict[str, object]] = []
        self.transfer_state = "DONE"
        self.check_states: list[str] = []
        self._next_handle = 10

    def get_agent_metadata(self) -> bytes:
        return b"local"

    def add_remote_agent(self, metadata: bytes) -> str | bytes:
        return metadata if self.names_as_bytes else metadata.decode()

    def remove_remote_agent(self, agent_name: str | bytes) -> None:
        pass

    def send_notif(self, agent_name: str | bytes, notif_msg: bytes) -> None:
        self.sent_notifications.append((agent_name, notif_msg))

    def get_new_notifs(self) -> dict[str | bytes, list[bytes]]:
        self.poll_calls += 1
        notifications = self.notifications
        self.notifications = {}
        return notifications

    def get_xfer_descs(self, descs, memory_type):
        assert memory_type == "VRAM"
        return descs

    def prep_xfer_dlist(self, agent_name, descs):
        handle = self._next_handle
        self._next_handle += 1
        return handle

    def make_prepped_xfer(
        self,
        operation,
        local_handle,
        local_indices,
        remote_handle,
        remote_indices,
        notif_msg=None,
    ) -> int:
        handle = self._next_handle
        self._next_handle += 1
        self.prepped_xfers.append(
            {
                "operation": operation,
                "local_handle": local_handle,
                "local_indices": local_indices,
                "remote_handle": remote_handle,
                "remote_indices": remote_indices,
                "notif_msg": notif_msg,
                "xfer_handle": handle,
            }
        )
        return handle

    def transfer(self, handle: int) -> str:
        self.transfer_calls.append(handle)
        return self.transfer_state

    def check_xfer_state(self, handle: int) -> str:
        self.check_calls.append(handle)
        if self.check_states:
            return self.check_states.pop(0)
        return self.transfer_state

    def release_xfer_handle(self, handle: int) -> None:
        self.released_xfers.append(handle)

    def release_dlist_handle(self, handle: int) -> None:
        self.released_dlists.append(handle)


def make_key(generation: int = 0, source: int = 0, reader: int = 1) -> TransferKey:
    return TransferKey(
        generation=generation, layer=LAYER, expert=EXPERT, source=source, reader=reader
    )


def make_tracker(rank: int) -> NixlEplbTransferTracker:
    return NixlEplbTransferTracker(rank, clock=FakeClock())


def agent_name(rank: int) -> str:
    return f"agent-{rank}"


def make_communicator(
    rank: int,
    agent: FakeNixlAgent,
    *,
    protocol: bool = True,
    clock: FakeClock | None = None,
    timeout_seconds: float = 300.0,
) -> NixlEplbCommunicator:
    """Build a communicator around a fake agent, bypassing NIXL init and the
    metadata collectives; mirrors the fields set in __init__."""
    peers = [peer for peer in range(WORLD_SIZE) if peer != rank]
    communicator = object.__new__(NixlEplbCommunicator)
    communicator._cpu_group = None
    communicator._rank = rank
    communicator._world_size = WORLD_SIZE
    communicator._num_local_experts = 1
    communicator._cuda_device_id = 0
    communicator._nixl_wrapper = agent
    communicator._nixl_memory_type = "VRAM"
    communicator._xfer_entries = []
    communicator._expert_to_src_row = None
    communicator._layer_idx = None
    communicator._protocol = protocol
    communicator._tracker = NixlEplbTransferTracker(
        rank, timeout_seconds=timeout_seconds, clock=clock or FakeClock()
    )
    communicator._generation = 0
    communicator._pending_reads = {}
    communicator._inflight = set()
    communicator._remote_agents = {peer: agent_name(peer) for peer in peers}
    communicator._remote_agent_ranks = {agent_name(peer): peer for peer in peers}
    communicator._remote_send_meta = {
        peer: {(LAYER, 0): (0x1000 * (peer + 1), TENSOR.nbytes, 0)} for peer in peers
    }
    return communicator


NO_OP_STREAM = SimpleNamespace(synchronize=lambda: None)


def begin_layer(
    communicator: NixlEplbCommunicator,
    *,
    layer: int = LAYER,
    stream: SimpleNamespace = NO_OP_STREAM,
) -> None:
    # The test host has no accelerator; stand in for the current stream.
    with mock.patch.object(torch.accelerator, "current_stream", return_value=stream):
        communicator.set_transfer_context(OLD_INDICES, layer)


def deliver(agent: FakeNixlAgent, sender: int, notification: NixlEplbNotification):
    agent.notifications.setdefault(agent_name(sender), []).append(notification.encode())


def corrupt(offset: int, value: int) -> bytes:
    payload = bytearray(NixlEplbNotification(READY, make_key()).encode())
    payload[offset] = value
    return bytes(payload)


@pytest.mark.parametrize("kind", [READY, READ_DONE])
def test_notification_round_trip(kind: int) -> None:
    notification = NixlEplbNotification(kind, make_key(generation=1 << 40))
    payload = notification.encode()
    assert len(payload) == 30
    assert NixlEplbNotification.decode(payload) == notification


def test_transfer_key_source_key_drops_reader() -> None:
    key = make_key()
    assert key.source_key == (key.generation, key.layer, key.expert, key.source)


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(b"", id="short"),
        pytest.param(
            NixlEplbNotification(READY, make_key()).encode() + b"\0", id="long"
        ),
        pytest.param(corrupt(0, ord("X")), id="magic"),
        pytest.param(corrupt(4, 2), id="version"),
        pytest.param(corrupt(5, 3), id="kind"),
    ],
)
def test_notification_decode_rejects_corrupt_payload(payload: bytes) -> None:
    with pytest.raises(ValueError):
        NixlEplbNotification.decode(payload)


def test_begin_generation_rejects_overlap_and_replay() -> None:
    tracker = make_tracker(rank=0)
    tracker.begin_generation(0)
    with pytest.raises(RuntimeError, match="still open"):
        tracker.begin_generation(1)
    tracker.end_generation(success=True)
    with pytest.raises(RuntimeError, match="does not follow"):
        tracker.begin_generation(0)
    tracker.begin_generation(1)
    assert tracker.active_generation == 1


def test_end_generation_requires_open_generation() -> None:
    with pytest.raises(RuntimeError, match="no open generation"):
        make_tracker(rank=0).end_generation(success=True)


def test_add_expected_reader_requires_open_generation() -> None:
    with pytest.raises(RuntimeError, match="outside the open generation"):
        make_tracker(rank=0).add_expected_reader(make_key().source_key, reader=1)


def test_take_ready_consumes_exact_key_once() -> None:
    tracker = make_tracker(rank=1)
    tracker.begin_generation(0)
    key = make_key()
    assert not tracker.take_ready(key)
    tracker.record_ready(key, sender=0)
    tracker.record_ready(key, sender=0)
    assert tracker.take_ready(key)
    assert not tracker.take_ready(key)
    assert not tracker.take_ready(key._replace(expert=12))


def test_ready_for_future_generation_is_held_until_it_begins() -> None:
    tracker = make_tracker(rank=1)
    tracker.begin_generation(0)
    future = make_key(generation=1)
    tracker.record_ready(future, sender=0)
    with pytest.raises(RuntimeError, match="outside the open generation"):
        tracker.take_ready(future)
    tracker.end_generation(success=True)
    tracker.begin_generation(1)
    assert tracker.take_ready(future)


def test_ready_for_closed_generation_raises() -> None:
    tracker = make_tracker(rank=1)
    tracker.begin_generation(0)
    tracker.end_generation(success=True)
    with pytest.raises(RuntimeError, match="closed generation"):
        tracker.record_ready(make_key(generation=0), sender=0)


@pytest.mark.parametrize("sender, rank", [(1, 1), (0, 0)])
def test_ready_rejects_wrong_route(sender: int, rank: int) -> None:
    tracker = make_tracker(rank=rank)
    tracker.begin_generation(0)
    with pytest.raises(RuntimeError, match="misrouted"):
        tracker.record_ready(make_key(source=0, reader=1), sender=sender)


def test_read_done_requires_declared_reader() -> None:
    tracker = make_tracker(rank=0)
    tracker.begin_generation(0)
    key = make_key(source=0, reader=1)
    with pytest.raises(RuntimeError, match="undeclared reader"):
        tracker.record_read_done(key, sender=1)
    tracker.add_expected_reader(key.source_key, reader=1)
    assert not tracker.sender_complete()
    tracker.record_read_done(key, sender=1)
    tracker.record_read_done(key, sender=1)
    assert tracker.sender_complete()


def test_read_done_is_strict_to_the_open_generation() -> None:
    tracker = make_tracker(rank=0)
    tracker.begin_generation(0)
    key = make_key(source=0, reader=1)
    tracker.add_expected_reader(key.source_key, reader=1)
    with pytest.raises(RuntimeError, match="outside the open generation"):
        tracker.record_read_done(key._replace(generation=1), sender=1)
    tracker.end_generation(success=False)
    with pytest.raises(RuntimeError, match="outside the open generation"):
        tracker.record_read_done(key, sender=1)


@pytest.mark.parametrize("sender, rank", [(0, 0), (1, 1)])
def test_read_done_rejects_wrong_route(sender: int, rank: int) -> None:
    tracker = make_tracker(rank=rank)
    tracker.begin_generation(0)
    with pytest.raises(RuntimeError, match="misrouted"):
        tracker.record_read_done(make_key(source=0, reader=1), sender=sender)


def test_sender_complete_with_nothing_declared() -> None:
    tracker = make_tracker(rank=0)
    tracker.begin_generation(0)
    assert tracker.sender_complete()


def test_end_generation_rejects_unconsumed_ready_then_closes_on_failure() -> None:
    tracker = make_tracker(rank=1)
    tracker.begin_generation(0)
    tracker.record_ready(make_key(), sender=0)
    with pytest.raises(RuntimeError, match="without matching add_recv"):
        tracker.end_generation(success=True)
    assert tracker.active_generation == 0
    tracker.end_generation(success=False)
    assert tracker.active_generation is None
    assert tracker.completed_generation == 0
    assert not tracker.ready


def test_end_generation_prunes_only_the_closed_generation() -> None:
    tracker = make_tracker(rank=1)
    tracker.begin_generation(0)
    sent = make_key(source=1, reader=0)
    tracker.add_expected_reader(sent.source_key, reader=0)
    tracker.record_read_done(sent, sender=0)
    future = make_key(generation=2)
    tracker.record_ready(future, sender=0)
    tracker.end_generation(success=True)
    assert set(tracker.ready) == {future}
    assert not tracker.expected and not tracker.completed


def test_expired_uses_injected_clock_and_timeout() -> None:
    clock = FakeClock()
    tracker = NixlEplbTransferTracker(rank=0, timeout_seconds=2.0, clock=clock)
    assert not tracker.expired()
    tracker.begin_generation(0)
    clock.now = 1.9
    assert not tracker.expired()
    clock.now = 2.0
    assert tracker.expired()


def test_set_transfer_context_synchronizes_stream_before_opening_generation() -> None:
    agent = FakeNixlAgent()
    communicator = make_communicator(rank=0, agent=agent)
    generations_at_sync: list[int | None] = []

    def synchronize() -> None:
        generations_at_sync.append(communicator._tracker.active_generation)

    recorder = SimpleNamespace(synchronize=synchronize)
    begin_layer(communicator, stream=recorder)
    assert generations_at_sync == [None]
    assert communicator._tracker.active_generation == 0

    legacy = make_communicator(rank=0, agent=agent, protocol=False)
    begin_layer(legacy, stream=recorder)
    assert generations_at_sync == [None]


def test_add_send_registers_reader_and_sends_ready() -> None:
    agent = FakeNixlAgent()
    sender = make_communicator(rank=0, agent=agent)
    begin_layer(sender)
    sender.add_send([TENSOR], dst_rank=1, expert_id=EXPERT)
    key = make_key()
    assert sender._tracker.expected == {key.source_key: {1}}
    assert agent.sent_notifications == [
        (agent_name(1), NixlEplbNotification(READY, key).encode())
    ]


def test_add_send_registers_reader_before_sending_ready(monkeypatch) -> None:
    agent = FakeNixlAgent()
    sender = make_communicator(rank=0, agent=agent)
    begin_layer(sender)

    def fail(*args, **kwargs):
        raise ConnectionError("link down")

    monkeypatch.setattr(agent, "send_notif", fail)
    with pytest.raises(ConnectionError):
        sender.add_send([TENSOR], dst_rank=1, expert_id=EXPERT)
    assert sender._tracker.expected == {make_key().source_key: {1}}


def test_add_recv_without_ready_queues_and_posts_nothing() -> None:
    agent = FakeNixlAgent()
    reader = make_communicator(rank=1, agent=agent)
    begin_layer(reader)
    reader.add_recv([TENSOR], src_rank=0, expert_id=EXPERT)
    assert agent.prepped_xfers == [] and agent.transfer_calls == []
    assert set(reader._pending_reads) == {make_key()}
    assert agent.poll_calls == 1


def test_add_recv_with_ready_posts_read_with_read_done_attached() -> None:
    agent = FakeNixlAgent()
    reader = make_communicator(rank=1, agent=agent)
    begin_layer(reader)
    deliver(agent, 0, NixlEplbNotification(READY, make_key()))
    reader.add_recv([TENSOR], src_rank=0, expert_id=EXPERT)
    (xfer,) = agent.prepped_xfers
    assert xfer["operation"] == "READ"
    assert xfer["notif_msg"] == NixlEplbNotification(READ_DONE, make_key()).encode()
    assert agent.transfer_calls == [xfer["xfer_handle"]]
    assert not reader._pending_reads


def test_execute_posts_pending_read_when_ready_arrives() -> None:
    agent = FakeNixlAgent()
    reader = make_communicator(rank=1, agent=agent)
    begin_layer(reader)
    reader.add_recv([TENSOR], src_rank=0, expert_id=EXPERT)
    deliver(agent, 0, NixlEplbNotification(READY, make_key()))
    reader.execute()
    (xfer,) = agent.prepped_xfers
    assert xfer["notif_msg"] == NixlEplbNotification(READ_DONE, make_key()).encode()
    assert agent.released_xfers == [xfer["xfer_handle"]]
    assert set(agent.released_dlists) == {xfer["local_handle"], xfer["remote_handle"]}
    assert reader._tracker.active_generation is None
    assert reader._tracker.completed_generation == 0
    assert reader._layer_idx is None and reader._expert_to_src_row is None


@pytest.mark.parametrize("ready_first", [True, False])
def test_duplicate_add_recv_raises(ready_first: bool) -> None:
    agent = FakeNixlAgent()
    reader = make_communicator(rank=1, agent=agent)
    begin_layer(reader)
    if ready_first:
        deliver(agent, 0, NixlEplbNotification(READY, make_key()))
    reader.add_recv([TENSOR], src_rank=0, expert_id=EXPERT)
    with pytest.raises(RuntimeError, match="duplicate add_recv"):
        reader.add_recv([TENSOR], src_rank=0, expert_id=EXPERT)


def test_execute_times_out_naming_the_missing_ready() -> None:
    agent = FakeNixlAgent()
    reader = make_communicator(
        rank=1, agent=agent, clock=FakeClock(step=0.5), timeout_seconds=1.0
    )
    begin_layer(reader)
    reader.add_recv([TENSOR], src_rank=0, expert_id=EXPERT)
    expected = re.escape(f"READY missing for [{make_key()!r}]")
    with pytest.raises(RuntimeError, match=expected):
        reader.execute()
    assert agent.prepped_xfers == []
    assert not reader._pending_reads
    assert reader._tracker.active_generation is None
    assert reader._tracker.completed_generation == 0


def test_execute_waits_for_read_done_from_every_declared_reader() -> None:
    agent = FakeNixlAgent()
    sender = make_communicator(
        rank=0, agent=agent, clock=FakeClock(step=0.5), timeout_seconds=1.0
    )
    begin_layer(sender)
    sender.add_send([TENSOR], dst_rank=1, expert_id=EXPERT)
    expected = re.escape(f"READ_DONE missing from {{{make_key().source_key!r}: [1]}}")
    with pytest.raises(RuntimeError, match=expected):
        sender.execute()
    assert sender._tracker.active_generation is None
    assert sender._tracker.completed_generation == 0
    deliver(agent, 1, NixlEplbNotification(READ_DONE, make_key()))
    with pytest.raises(RuntimeError, match="outside the open generation"):
        sender._drain_notifications()


def test_execute_returns_once_every_read_done_arrived() -> None:
    agent = FakeNixlAgent()
    sender = make_communicator(rank=0, agent=agent)
    begin_layer(sender)
    sender.add_send([TENSOR], dst_rank=1, expert_id=EXPERT)
    deliver(agent, 1, NixlEplbNotification(READ_DONE, make_key()))
    sender.execute()
    assert sender._tracker.completed_generation == 0
    assert agent.released_xfers == []


def test_execute_with_nothing_enqueued_drains_once_and_skips_barrier(
    monkeypatch,
) -> None:
    agent = FakeNixlAgent()
    communicator = make_communicator(rank=0, agent=agent)
    monkeypatch.setattr(
        communicator, "_post_read_barrier", lambda: pytest.fail("barrier ran")
    )
    for layer in range(2):
        begin_layer(communicator, layer=layer)
        communicator.execute()
    assert agent.poll_calls == 2
    assert communicator._tracker.completed_generation == 1
    assert communicator._generation == 2


def test_early_ready_is_consumed_in_its_own_generation() -> None:
    agent = FakeNixlAgent()
    reader = make_communicator(rank=1, agent=agent)
    begin_layer(reader)
    deliver(agent, 0, NixlEplbNotification(READY, make_key(generation=1)))
    reader.execute()
    assert reader._tracker.completed_generation == 0
    begin_layer(reader)
    reader.add_recv([TENSOR], src_rank=0, expert_id=EXPERT)
    assert len(agent.prepped_xfers) == 1
    assert not reader._pending_reads


def test_execute_rejects_ready_without_matching_add_recv() -> None:
    agent = FakeNixlAgent()
    reader = make_communicator(rank=1, agent=agent)
    begin_layer(reader)
    deliver(agent, 0, NixlEplbNotification(READY, make_key()))
    with pytest.raises(RuntimeError, match="without matching add_recv"):
        reader.execute()
    assert reader._tracker.active_generation is None
    assert reader._tracker.completed_generation == 0


def test_failed_read_releases_handles_and_closes_generation() -> None:
    agent = FakeNixlAgent()
    agent.transfer_state = "PROC"
    agent.check_states = ["ERR"]
    reader = make_communicator(rank=1, agent=agent)
    begin_layer(reader)
    deliver(agent, 0, NixlEplbNotification(READY, make_key()))
    reader.add_recv([TENSOR], src_rank=0, expert_id=EXPERT)
    (xfer,) = agent.prepped_xfers
    assert reader._inflight == {xfer["xfer_handle"]}
    with pytest.raises(RuntimeError, match="NIXL transfer failed"):
        reader.execute()
    assert agent.released_xfers == [xfer["xfer_handle"]]
    assert not reader._inflight and not reader._xfer_entries
    assert reader._tracker.active_generation is None
    deliver(agent, 0, NixlEplbNotification(READY, make_key()))
    with pytest.raises(RuntimeError, match="closed generation"):
        reader._drain_notifications()


def test_notification_from_unknown_agent_raises() -> None:
    agent = FakeNixlAgent()
    reader = make_communicator(rank=1, agent=agent)
    begin_layer(reader)
    agent.notifications = {
        "stranger": [NixlEplbNotification(READY, make_key()).encode()]
    }
    with pytest.raises(RuntimeError, match="unknown agent"):
        reader.execute()


def test_two_ranks_complete_one_layer_without_a_barrier(monkeypatch) -> None:
    agent0, agent1 = FakeNixlAgent(), FakeNixlAgent()
    sender = make_communicator(rank=0, agent=agent0)
    reader = make_communicator(rank=1, agent=agent1)
    for communicator in (sender, reader):
        monkeypatch.setattr(
            communicator, "_post_read_barrier", lambda: pytest.fail("barrier ran")
        )
        begin_layer(communicator)

    sender.add_send([TENSOR], dst_rank=1, expert_id=EXPERT)
    ((target, ready),) = agent0.sent_notifications
    assert target == agent_name(1)
    agent1.notifications[agent_name(0)] = [ready]

    reader.add_recv([TENSOR], src_rank=0, expert_id=EXPERT)
    reader.execute()
    (xfer,) = agent1.prepped_xfers
    read_done = xfer["notif_msg"]
    assert isinstance(read_done, bytes)
    agent0.notifications[agent_name(1)] = [read_done]

    sender.execute()
    assert sender._tracker.completed_generation == 0
    assert reader._tracker.completed_generation == 0


def test_legacy_mode_keeps_main_behaviour(monkeypatch) -> None:
    agent = FakeNixlAgent()
    communicator = make_communicator(rank=1, agent=agent, protocol=False)
    barriers: list[int] = []
    monkeypatch.setattr(communicator, "_post_read_barrier", lambda: barriers.append(1))
    begin_layer(communicator)
    communicator.add_send([TENSOR], dst_rank=0, expert_id=12)
    communicator.add_recv([TENSOR], src_rank=0, expert_id=EXPERT)
    (xfer,) = agent.prepped_xfers
    assert xfer["notif_msg"] == b""
    assert agent.transfer_calls == [xfer["xfer_handle"]]
    communicator.execute()
    assert barriers == [1]
    assert agent.sent_notifications == [] and agent.poll_calls == 0
    assert communicator._tracker.active_generation is None
    assert communicator._generation == 0
    assert agent.released_xfers == [xfer["xfer_handle"]]


@pytest.mark.parametrize("names_as_bytes", [False, True])
def test_init_remote_agents_maps_agent_names_to_ranks(
    monkeypatch, names_as_bytes: bool
) -> None:
    agent = FakeNixlAgent(names_as_bytes=names_as_bytes)
    communicator = make_communicator(rank=0, agent=agent)
    communicator._remote_agents = {}
    communicator._remote_agent_ranks = {}

    def fake_all_gather_object(out, obj, group=None):
        out[:] = [b"agent-0", b"agent-1"]

    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)
    communicator._init_remote_agents()
    assert communicator._remote_agents == {
        1: b"agent-1" if names_as_bytes else "agent-1"
    }
    assert communicator._remote_agent_ranks == {"agent-1": 1}


@pytest.mark.parametrize("stateless", [False, True])
def test_factory_forwards_protocol_flag_only_for_regular_groups(
    monkeypatch, stateless: bool
) -> None:
    created: dict[str, object] = {}

    class Recorder:
        def __init__(self, **kwargs) -> None:
            created.update(kwargs)

    monkeypatch.setattr(eplb_communicator, "has_nixl", lambda: True)
    monkeypatch.setattr(eplb_communicator, "NixlEplbCommunicator", Recorder)
    monkeypatch.setattr(
        eplb_communicator,
        "current_platform",
        SimpleNamespace(is_cuda_alike=lambda: True),
    )
    group = (
        object.__new__(StatelessGroupCoordinator) if stateless else SimpleNamespace()
    )
    group.cpu_group = None
    group.device_group = None
    weight = SimpleNamespace(device=SimpleNamespace(type="cuda"))

    create_eplb_communicator(
        group, "nixl", [[weight]], [weight], enable_nixl_sync_protocol=True
    )
    assert created["enable_sync_protocol"] is (not stateless)
