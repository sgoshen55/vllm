# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import pytest
import torch

from vllm.distributed.eplb.eplb_communicator import (
    _NIXL_EPLB_NOTIFICATION_STRUCT,
    NixlEplbCommunicator,
    _NixlEplbDeadlinePhase,
    _NixlEplbNotification,
    _NixlEplbNotificationDisposition,
    _NixlEplbNotificationKind,
    _NixlEplbProtocolState,
    _NixlEplbTransferKey,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeNixlAgent:
    def get_agent_metadata(self) -> bytes:
        return b"local"

    def add_remote_agent(self, metadata: bytes) -> str:
        return metadata.decode()

    def remove_remote_agent(self, agent_name: str) -> None:
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


def test_remote_agent_names_map_back_to_ranks(monkeypatch: pytest.MonkeyPatch) -> None:
    communicator = object.__new__(NixlEplbCommunicator)
    communicator._rank = 0
    communicator._world_size = 3
    communicator._cpu_group = object()
    communicator._nixl_wrapper = FakeNixlAgent()
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

    assert communicator._remote_agents == {1: "peer-1", 2: "peer-2"}
    assert communicator._remote_agent_ranks == {"peer-1": 1, "peer-2": 2}


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
