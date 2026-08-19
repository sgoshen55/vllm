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

    def get_agent_metadata(self) -> bytes:
        return b"local"

    def add_remote_agent(self, metadata: bytes) -> str | bytes:
        return metadata if self.names_as_bytes else metadata.decode()

    def remove_remote_agent(self, agent_name: str | bytes) -> None:
        pass

    def release_xfer_handle(self, handle: int) -> None:
        self.released_xfers.append(handle)

    def release_dlist_handle(self, handle: int) -> None:
        self.released_dlists.append(handle)


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


def test_legacy_execute_records_baseline_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    agent = FakeNixlAgent()
    communicator = object.__new__(NixlEplbCommunicator)
    communicator._rank = 0
    communicator._sync_protocol_active = True
    communicator._xfer_entries = [(10, 20, 30)]
    communicator._pending_read_post_seconds = 1.5
    communicator._expert_to_src_row = [{}]
    communicator._layer_idx = 4
    communicator._nixl_wrapper = agent
    events = []

    def wait_for_all_transfers(handles: list[int]) -> None:
        events.append(("wait", handles))
        clock.advance(2)

    def post_read_barrier() -> None:
        events.append(("barrier", None))
        clock.advance(3)

    monkeypatch.setattr(eplb_communicator.time, "perf_counter", clock)
    monkeypatch.setattr(
        communicator,
        "_wait_for_all_transfers",
        wait_for_all_transfers,
    )
    monkeypatch.setattr(communicator, "_post_read_barrier", post_read_barrier)

    communicator.execute()

    stats = communicator._last_execute_stats
    assert stats is not None
    assert stats.sync_protocol_active
    assert stats.reads_posted == 1
    assert stats.read_post_seconds == 1.5
    assert stats.read_progress_seconds == 2
    assert stats.barrier_seconds == 3
    assert stats.execute_seconds == 5
    assert stats.ready_sent == 0
    assert stats.ready_received == 0
    assert stats.read_done_attached == 0
    assert stats.read_done_received == 0
    assert events == [("wait", [30]), ("barrier", None)]
    assert agent.released_xfers == [30]
    assert agent.released_dlists == [10, 20]
    assert communicator._xfer_entries == []
    assert communicator._pending_read_post_seconds == 0


def test_non_sync_execute_keeps_legacy_path_uninstrumented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    communicator = object.__new__(NixlEplbCommunicator)
    communicator._sync_protocol_active = False
    communicator._xfer_entries = []
    communicator._pending_read_post_seconds = 0.0
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
