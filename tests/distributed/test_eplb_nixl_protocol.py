# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import vllm.distributed.eplb.eplb_communicator as eplb_communicator
from vllm.distributed.eplb.eplb_communicator import (
    _NIXL_EPLB_NOTIFICATION_STRUCT,
    NixlEplbCommunicator,
    _NixlEplbDeadlinePhase,
    _NixlEplbNotification,
    _NixlEplbNotificationDisposition,
    _NixlEplbNotificationKind,
    _NixlEplbProtocolState,
    _NixlEplbTransferKey,
    create_eplb_communicator,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


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


def make_fake_tensor(address: int = 2000, nbytes: int = 16):
    return SimpleNamespace(nbytes=nbytes, data_ptr=lambda: address)


def make_sync_communicator(
    *,
    rank: int,
    agent: FakeNixlAgent,
    generation: int = 7,
    layer: int = 3,
) -> NixlEplbCommunicator:
    peer = 1 - rank
    communicator = object.__new__(NixlEplbCommunicator)
    communicator._rank = rank
    communicator._world_size = 2
    communicator._sync_protocol_active = True
    communicator._sync_protocol_state = _NixlEplbProtocolState(
        local_rank=rank,
        world_size=2,
    )
    communicator._sync_protocol_state.begin_generation(generation)
    communicator._next_sync_generation = generation + 1
    communicator._sync_stats = eplb_communicator._NixlEplbExecuteStats(
        sync_protocol_active=True
    )
    communicator._xfer_entries = []
    communicator._pending_reads = {}
    communicator._active_reads = {}
    communicator._recv_keys = set()
    communicator._source_readers = {}
    communicator._source_expectations_frozen = False
    communicator._ready_sent_at = {}
    communicator._deferred_notifications = {}
    communicator._last_execute_stats = None
    communicator._expert_to_src_row = [{11: 0}, {11: 0}]
    communicator._layer_idx = layer
    communicator._nixl_wrapper = agent
    communicator._nixl_memory_type = "VRAM"
    communicator._cuda_device_id = rank
    communicator._remote_agents = {peer: f"peer-{peer}"}
    communicator._remote_agent_ranks = {f"peer-{peer}": peer}
    communicator._remote_send_meta = {
        peer: {(layer, 0): (1000, 16, peer)},
    }
    communicator._registered_descs = []
    return communicator


class FakeGroupCoordinator:
    cpu_group = object()
    device_group = object()
    device_communicator = None


class FakeStatelessGroupCoordinator(FakeGroupCoordinator):
    pass


def make_transfer_key(
    *,
    generation: int = 7,
    layer: int = 3,
    expert: int = 11,
    source: int = 0,
    reader: int = 1,
) -> _NixlEplbTransferKey:
    return _NixlEplbTransferKey(
        generation=generation,
        layer=layer,
        expert=expert,
        source=source,
        reader=reader,
        tensor_group=0,
    )


@pytest.mark.parametrize("kind", list(_NixlEplbNotificationKind))
def test_notification_round_trip_preserves_full_identity(
    kind: _NixlEplbNotificationKind,
) -> None:
    notification = _NixlEplbNotification(kind, make_transfer_key())

    payload = notification.encode()

    assert len(payload) == _NIXL_EPLB_NOTIFICATION_STRUCT.size
    assert _NixlEplbNotification.decode(payload) == notification


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda payload: payload[:-1], "notification length"),
        (lambda payload: b"FAIL" + payload[4:], "notification magic"),
        (
            lambda payload: payload[:4] + b"\x02" + payload[5:],
            "notification version",
        ),
        (
            lambda payload: payload[:5] + b"\xff" + payload[6:],
            "notification kind",
        ),
    ],
)
def test_notification_decode_rejects_malformed_payloads(mutate, error: str) -> None:
    payload = _NixlEplbNotification(
        _NixlEplbNotificationKind.READY,
        make_transfer_key(),
    ).encode()

    with pytest.raises(ValueError, match=error):
        _NixlEplbNotification.decode(mutate(payload))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("generation", -1),
        ("generation", 1 << 64),
        ("layer", 1 << 32),
        ("expert", -1),
        ("source", 1 << 32),
        ("reader", -1),
        ("tensor_group", 1 << 32),
    ],
)
def test_notification_encode_rejects_out_of_range_identity(
    field: str,
    value: int,
) -> None:
    key = replace(make_transfer_key(), **{field: value})
    notification = _NixlEplbNotification(
        _NixlEplbNotificationKind.READY,
        key,
    )

    with pytest.raises(ValueError, match=field):
        notification.encode()


@pytest.mark.parametrize(
    ("kind", "sender", "receiver"),
    [
        (_NixlEplbNotificationKind.READY, 0, 1),
        (_NixlEplbNotificationKind.READ_DONE, 1, 0),
    ],
)
def test_notification_route_matches_kind(
    kind: _NixlEplbNotificationKind,
    sender: int,
    receiver: int,
) -> None:
    notification = _NixlEplbNotification(kind, make_transfer_key())

    notification.validate_route(sender, receiver)

    with pytest.raises(RuntimeError, match="route mismatch"):
        notification.validate_route(receiver, sender)


@pytest.mark.parametrize("names_as_bytes", [False, True])
def test_remote_agent_names_map_back_to_ranks(
    monkeypatch: pytest.MonkeyPatch,
    names_as_bytes: bool,
) -> None:
    communicator = object.__new__(NixlEplbCommunicator)
    communicator._rank = 0
    communicator._world_size = 3
    communicator._cpu_group = object()
    communicator._nixl_wrapper = FakeNixlAgent(names_as_bytes=names_as_bytes)
    communicator._remote_agents = {}
    communicator._remote_agent_ranks = {}

    def fake_all_gather_object(output, value, group) -> None:
        assert value == b"local"
        assert group is communicator._cpu_group
        output[:] = [b"local", b"peer-1", b"peer-2"]

    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        fake_all_gather_object,
    )

    communicator._init_remote_agents()

    expected_handles = (
        {1: b"peer-1", 2: b"peer-2"} if names_as_bytes else {1: "peer-1", 2: "peer-2"}
    )
    assert communicator._remote_agents == expected_handles
    assert communicator._remote_agent_ranks == {"peer-1": 1, "peer-2": 2}


@pytest.mark.parametrize(
    ("requested", "stateless", "expected"),
    [
        (True, False, True),
        (False, False, False),
        (True, True, False),
    ],
)
def test_sync_protocol_activation_is_requested_and_non_elastic(
    monkeypatch: pytest.MonkeyPatch,
    requested: bool,
    stateless: bool,
    expected: bool,
) -> None:
    monkeypatch.setattr(
        eplb_communicator,
        "StatelessGroupCoordinator",
        FakeStatelessGroupCoordinator,
    )
    monkeypatch.setattr(eplb_communicator, "has_nixl", lambda: True)
    monkeypatch.setattr(
        eplb_communicator,
        "current_platform",
        SimpleNamespace(is_cuda_alike=lambda: True),
    )
    constructor_args = {}

    def fake_nixl_communicator(**kwargs):
        constructor_args.update(kwargs)
        return object()

    monkeypatch.setattr(
        eplb_communicator,
        "NixlEplbCommunicator",
        fake_nixl_communicator,
    )
    coordinator = (
        FakeStatelessGroupCoordinator() if stateless else FakeGroupCoordinator()
    )
    cuda_tensor = SimpleNamespace(device=SimpleNamespace(type="cuda"))

    create_eplb_communicator(
        group_coordinator=coordinator,  # type: ignore[arg-type]
        backend="nixl",
        expert_weights=[[cuda_tensor]],  # type: ignore[list-item]
        expert_buffer=[cuda_tensor],  # type: ignore[list-item]
        enable_nixl_sync_protocol=requested,
    )

    assert constructor_args["defer_remote_setup"] is stateless
    assert constructor_args["enable_sync_protocol"] is expected


def test_add_send_publishes_ready_immediately() -> None:
    agent = FakeNixlAgent()
    communicator = make_sync_communicator(rank=0, agent=agent)

    communicator.add_send([], dst_rank=1, expert_id=11)

    assert len(agent.sent_notifications) == 1
    peer, payload = agent.sent_notifications[0]
    assert peer == "peer-1"
    assert _NixlEplbNotification.decode(payload) == _NixlEplbNotification(
        _NixlEplbNotificationKind.READY,
        make_transfer_key(),
    )
    assert agent.poll_calls == 0
    assert agent.transfer_calls == []
    assert communicator._sync_stats is not None
    assert communicator._sync_stats.ready_sent == 1


def test_add_recv_polls_once_and_queues_when_ready_is_absent() -> None:
    agent = FakeNixlAgent()
    communicator = make_sync_communicator(rank=1, agent=agent)

    communicator.add_recv(
        [make_fake_tensor()],  # type: ignore[list-item]
        src_rank=0,
        expert_id=11,
    )

    assert set(communicator._pending_reads) == {make_transfer_key()}
    assert communicator._active_reads == {}
    assert agent.poll_calls == 1
    assert agent.transfer_calls == []
    assert agent.check_calls == []


def test_add_recv_posts_read_without_waiting_when_ready_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = FakeNixlAgent()
    communicator = make_sync_communicator(rank=1, agent=agent)
    ready = _NixlEplbNotification(
        _NixlEplbNotificationKind.READY,
        make_transfer_key(),
    )
    agent.notifications = {"peer-0": [ready.encode()]}
    monkeypatch.setattr(
        communicator,
        "_post_read_barrier",
        lambda: pytest.fail("synchronous notification protocol must not barrier"),
    )

    communicator.add_recv(
        [make_fake_tensor()],  # type: ignore[list-item]
        src_rank=0,
        expert_id=11,
    )

    assert communicator._pending_reads == {}
    assert set(communicator._active_reads) == {make_transfer_key()}
    assert agent.poll_calls == 1
    assert len(agent.transfer_calls) == 1
    assert agent.check_calls == []
    read_done_payload = agent.prepped_xfers[0]["notif_msg"]
    assert isinstance(read_done_payload, bytes)
    assert _NixlEplbNotification.decode(read_done_payload) == _NixlEplbNotification(
        _NixlEplbNotificationKind.READ_DONE,
        make_transfer_key(),
    )

    communicator.execute()

    stats = communicator._last_execute_stats
    assert stats is not None
    assert stats.reads_posted == 1
    assert stats.ready_received == 1
    assert stats.read_done_attached == 1
    assert stats.barrier_seconds == 0
    assert agent.check_calls == []


def test_execute_matches_late_ready_and_completes_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = FakeNixlAgent()
    communicator = make_sync_communicator(rank=1, agent=agent)
    communicator.add_recv(
        [make_fake_tensor()],  # type: ignore[list-item]
        src_rank=0,
        expert_id=11,
    )
    ready = _NixlEplbNotification(
        _NixlEplbNotificationKind.READY,
        make_transfer_key(),
    )
    agent.notifications = {"peer-0": [ready.encode()]}
    monkeypatch.setattr(
        communicator,
        "_post_read_barrier",
        lambda: pytest.fail("synchronous notification protocol must not barrier"),
    )

    communicator.execute()

    assert communicator._last_execute_stats is not None
    assert communicator._last_execute_stats.reads_posted == 1
    assert communicator._last_execute_stats.notification_poll_calls == 2
    assert len(agent.transfer_calls) == 1


def test_read_completion_is_progressed_only_by_execute() -> None:
    agent = FakeNixlAgent()
    agent.transfer_state = "PROC"
    agent.check_states = ["PROC", "DONE"]
    communicator = make_sync_communicator(rank=1, agent=agent)
    ready = _NixlEplbNotification(
        _NixlEplbNotificationKind.READY,
        make_transfer_key(),
    )
    agent.notifications = {"peer-0": [ready.encode()]}

    communicator.add_recv(
        [make_fake_tensor()],  # type: ignore[list-item]
        src_rank=0,
        expert_id=11,
    )

    assert agent.check_calls == []

    communicator.execute()

    assert len(agent.check_calls) == 2
    assert communicator._last_execute_stats is not None
    assert communicator._last_execute_stats.read_completion_sum_seconds > 0


def test_future_ready_is_deferred_until_its_generation() -> None:
    agent = FakeNixlAgent()
    communicator = make_sync_communicator(rank=1, agent=agent)
    future_key = make_transfer_key(generation=8, layer=4)
    future_ready = _NixlEplbNotification(
        _NixlEplbNotificationKind.READY,
        future_key,
    )
    agent.notifications = {"peer-0": [future_ready.encode()]}

    communicator.execute()

    assert communicator._deferred_notifications == {
        8: [(future_ready, 0)],
    }

    communicator._layer_idx = 4
    communicator._expert_to_src_row = [{11: 0}, {11: 0}]
    communicator._remote_send_meta[0][(4, 0)] = (1000, 16, 0)
    communicator._sync_protocol_state.begin_generation(8)
    communicator._sync_stats = eplb_communicator._NixlEplbExecuteStats(
        sync_protocol_active=True
    )
    communicator.add_recv(
        [make_fake_tensor()],  # type: ignore[list-item]
        src_rank=0,
        expert_id=11,
    )

    assert communicator._deferred_notifications == {}
    assert set(communicator._active_reads) == {future_key}
    communicator.execute()


def test_execute_waits_for_every_expected_read_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = FakeNixlAgent()
    communicator = make_sync_communicator(rank=0, agent=agent)
    communicator.add_send([], dst_rank=1, expert_id=11)
    read_done = _NixlEplbNotification(
        _NixlEplbNotificationKind.READ_DONE,
        make_transfer_key(),
    )
    agent.notifications = {"peer-1": [read_done.encode()]}
    monkeypatch.setattr(
        communicator,
        "_post_read_barrier",
        lambda: pytest.fail("synchronous notification protocol must not barrier"),
    )

    communicator.execute()

    stats = communicator._last_execute_stats
    assert stats is not None
    assert stats.ready_sent == 1
    assert stats.read_done_received == 1
    assert stats.barrier_seconds == 0


def test_execute_times_out_when_ready_never_arrives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    agent = FakeNixlAgent()
    communicator = make_sync_communicator(rank=1, agent=agent)
    communicator._sync_protocol_state._clock = clock
    communicator._sync_protocol_state.timeout_seconds = 1
    communicator.add_recv(
        [make_fake_tensor()],  # type: ignore[list-item]
        src_rank=0,
        expert_id=11,
    )
    monkeypatch.setattr(eplb_communicator.time, "sleep", clock.advance)

    with pytest.raises(RuntimeError, match="missing READY"):
        communicator.execute()

    assert communicator._sync_protocol_state.active_generation is None
    assert communicator._pending_reads == {}


def test_non_sync_execute_keeps_legacy_path_uninstrumented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    communicator = object.__new__(NixlEplbCommunicator)
    communicator._sync_protocol_active = False
    communicator._xfer_entries = []
    communicator._pending_reads = {}
    communicator._active_reads = {}
    communicator._recv_keys = set()
    communicator._source_readers = {}
    communicator._ready_sent_at = {}
    communicator._source_expectations_frozen = False
    communicator._sync_stats = None
    communicator._last_execute_stats = None
    communicator._expert_to_src_row = None
    communicator._layer_idx = None
    communicator._nixl_wrapper = FakeNixlAgent()
    events = []

    monkeypatch.setattr(
        communicator,
        "_wait_for_all_transfers",
        lambda handles: events.append(("wait", handles)),
    )
    monkeypatch.setattr(
        communicator,
        "_post_read_barrier",
        lambda: events.append(("barrier", None)),
    )
    monkeypatch.setattr(
        eplb_communicator.time,
        "perf_counter",
        lambda: pytest.fail("non-sync path must not record protocol timings"),
    )

    communicator.execute()

    assert events == [("wait", []), ("barrier", None)]
    assert communicator._last_execute_stats is None


def test_non_sync_add_recv_keeps_legacy_transfer_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = FakeNixlAgent()
    communicator = make_sync_communicator(rank=1, agent=agent)
    communicator._sync_protocol_active = False
    communicator._sync_protocol_state.end_generation(success=False)
    communicator._sync_stats = None
    barriers = []
    monkeypatch.setattr(
        communicator,
        "_post_read_barrier",
        lambda: barriers.append(True),
    )

    communicator.add_recv(
        [make_fake_tensor()],  # type: ignore[list-item]
        src_rank=0,
        expert_id=11,
    )

    assert agent.poll_calls == 0
    assert agent.sent_notifications == []
    assert len(agent.transfer_calls) == 1
    assert agent.prepped_xfers[0]["notif_msg"] is None

    communicator.execute()

    assert barriers == [True]
    assert communicator._last_execute_stats is None


def test_expected_readers_are_frozen_and_deduplicated() -> None:
    state = _NixlEplbProtocolState(local_rank=0, world_size=4)
    state.begin_generation(7)
    source_key = make_transfer_key().source_key

    assert state.set_expected_readers(source_key, [2, 1, 2])
    assert not state.set_expected_readers(source_key, [1, 2])
    assert state.expected_readers(source_key) == frozenset({1, 2})
    assert not state.source_complete(source_key)

    with pytest.raises(RuntimeError, match="already frozen"):
        state.set_expected_readers(source_key, [1, 3])


def test_source_completes_only_after_every_expected_reader() -> None:
    state = _NixlEplbProtocolState(local_rank=0, world_size=3)
    state.begin_generation(7)
    source_key = make_transfer_key().source_key
    state.set_expected_readers(source_key, [1, 2])

    first = _NixlEplbNotification(
        _NixlEplbNotificationKind.READ_DONE,
        make_transfer_key(reader=1),
    )
    second = _NixlEplbNotification(
        _NixlEplbNotificationKind.READ_DONE,
        make_transfer_key(reader=2),
    )

    assert (
        state.record_notification(first, sender_rank=1)
        == _NixlEplbNotificationDisposition.RECORDED
    )
    assert not state.source_complete(source_key)
    assert (
        state.record_notification(first, sender_rank=1)
        == _NixlEplbNotificationDisposition.DUPLICATE
    )
    assert state.completed_readers(source_key) == frozenset({1})

    assert (
        state.record_notification(second, sender_rank=2)
        == _NixlEplbNotificationDisposition.RECORDED
    )
    assert state.source_complete(source_key)


def test_source_with_zero_readers_is_immediately_complete() -> None:
    state = _NixlEplbProtocolState(local_rank=0, world_size=2)
    state.begin_generation(7)
    source_key = make_transfer_key().source_key

    state.set_expected_readers(source_key, [])

    assert state.source_complete(source_key)


def test_ready_is_idempotent_and_consumed_once() -> None:
    state = _NixlEplbProtocolState(local_rank=1, world_size=2)
    state.begin_generation(7)
    key = make_transfer_key()
    ready = _NixlEplbNotification(_NixlEplbNotificationKind.READY, key)

    assert (
        state.record_notification(ready, sender_rank=0)
        == _NixlEplbNotificationDisposition.RECORDED
    )
    assert (
        state.record_notification(ready, sender_rank=0)
        == _NixlEplbNotificationDisposition.DUPLICATE
    )
    assert state.consume_ready(key)
    assert not state.consume_ready(key)
    assert (
        state.record_notification(ready, sender_rank=0)
        == _NixlEplbNotificationDisposition.DUPLICATE
    )
    assert not state.consume_ready(key)


@pytest.mark.parametrize(
    ("kind", "local_rank", "sender_rank"),
    [
        (_NixlEplbNotificationKind.READY, 1, 0),
        (_NixlEplbNotificationKind.READ_DONE, 0, 1),
    ],
)
def test_completed_generation_notification_is_stale(
    kind: _NixlEplbNotificationKind,
    local_rank: int,
    sender_rank: int,
) -> None:
    state = _NixlEplbProtocolState(local_rank=local_rank, world_size=2)
    state.begin_generation(7)
    state.end_generation(success=True)
    state.begin_generation(8)
    notification = _NixlEplbNotification(
        kind,
        make_transfer_key(generation=7),
    )

    assert (
        state.record_notification(notification, sender_rank=sender_rank)
        == _NixlEplbNotificationDisposition.STALE
    )


def test_unknown_current_generation_read_done_is_rejected() -> None:
    state = _NixlEplbProtocolState(local_rank=0, world_size=2)
    state.begin_generation(7)
    read_done = _NixlEplbNotification(
        _NixlEplbNotificationKind.READ_DONE,
        make_transfer_key(),
    )

    with pytest.raises(RuntimeError, match="Unknown.*READ_DONE"):
        state.record_notification(read_done, sender_rank=1)


def test_sender_identity_mismatch_is_rejected() -> None:
    state = _NixlEplbProtocolState(local_rank=1, world_size=3)
    state.begin_generation(7)
    ready = _NixlEplbNotification(
        _NixlEplbNotificationKind.READY,
        make_transfer_key(),
    )

    with pytest.raises(RuntimeError, match="route mismatch"):
        state.record_notification(ready, sender_rank=2)


def test_deadline_uses_injected_clock() -> None:
    clock = FakeClock()
    state = _NixlEplbProtocolState(
        local_rank=0,
        world_size=2,
        timeout_seconds=5,
        clock=clock,
    )
    state.begin_generation(7)
    source_key = make_transfer_key().source_key

    deadline = state.arm_deadline(_NixlEplbDeadlinePhase.READ_DONE, source_key)
    assert deadline.expires_at == 5
    assert state.expired_deadlines() == ()

    clock.advance(5)
    assert state.expired_deadlines() == (deadline,)

    state.clear_deadline(_NixlEplbDeadlinePhase.READ_DONE, source_key)
    assert state.expired_deadlines() == ()
