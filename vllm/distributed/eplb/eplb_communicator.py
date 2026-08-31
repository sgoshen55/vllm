# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
EPLB communicator implementations and factory.
"""

import contextlib
import struct
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum, IntEnum

import numpy as np
import torch
from torch.distributed import (
    P2POp,
    ProcessGroup,
    batch_isend_irecv,
)

import vllm.distributed.nixl_utils as nixl_utils
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.device_communicators.pynccl_wrapper import (
    ncclDataTypeEnum,
)
from vllm.distributed.parallel_state import (
    GroupCoordinator,
    get_pp_group,
    is_local_first_rank,
)
from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator
from vllm.distributed.utils import is_weak_contiguous
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

_NIXL_EPLB_NOTIFICATION_MAGIC = b"EPLB"
_NIXL_EPLB_NOTIFICATION_VERSION = 2
_NIXL_EPLB_NOTIFICATION_HEADER_STRUCT = struct.Struct("!4sBB")
_NIXL_EPLB_NOTIFICATION_STRUCT = struct.Struct("!4sBBQIIIII")
_NIXL_EPLB_ABORT_NOTIFICATION_STRUCT = struct.Struct("!4sBBQIIII")
_NIXL_EPLB_DEFAULT_TIMEOUT_SECONDS = 300.0


@dataclass(frozen=True, slots=True)
class EplbPerfContext:
    """Stable identifiers for one rank-local EPLB layer transfer."""

    rearrangement_id: int
    generation_id: int
    layer_id: int
    rank: int


@dataclass(slots=True)
class EplbBackendPerf:
    """Backend timing fields captured by one communicator execution."""

    fields: dict[str, int | float | str] = field(default_factory=dict)
    gpu_events: tuple[torch.cuda.Event, torch.cuda.Event] | None = None
    read_payloads: list[tuple[int, int, int]] = field(default_factory=list)


def _normalize_nixl_agent_name(agent_name: str | bytes) -> str:
    if isinstance(agent_name, bytes):
        return agent_name.decode()
    return agent_name


class _NixlEplbNotificationKind(IntEnum):
    READY = 1
    READ_DONE = 2
    ABORT = 3


class _NixlEplbAbortReason(IntEnum):
    PROTOCOL_ERROR = 1
    READY_TIMEOUT = 2
    READ_FAILURE = 3
    READ_TIMEOUT = 4
    READ_DONE_TIMEOUT = 5


class _NixlEplbNotificationDisposition(Enum):
    RECORDED = "recorded"
    DUPLICATE = "duplicate"
    STALE = "stale"


class _NixlEplbDeadlinePhase(Enum):
    CUDA_READINESS = "CUDA readiness"
    READY = "missing READY"
    READ = "local READ completion"
    READ_DONE = "missing READ_DONE"


class _NixlEplbPerfPhase(Enum):
    READY_WAIT = "ready_wait_ms"
    READ_EXECUTION = "read_execution_ms"
    READ_DONE_WAIT = "read_done_wait_ms"
    PROTOCOL_RESIDUAL = "protocol_residual_ms"


class _NixlEplbExclusivePhaseTimer:
    """Accumulate mutually exclusive protocol-state occupancy."""

    def __init__(
        self,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._clock = clock
        self._phase = _NixlEplbPerfPhase.PROTOCOL_RESIDUAL
        self._last_transition = clock()
        self._totals = dict.fromkeys(_NixlEplbPerfPhase, 0.0)

    def transition(self, phase: _NixlEplbPerfPhase) -> None:
        now = self._clock()
        self._totals[self._phase] += now - self._last_transition
        self._phase = phase
        self._last_transition = now

    def finish(self) -> dict[_NixlEplbPerfPhase, float]:
        self.transition(self._phase)
        return dict(self._totals)


@dataclass(frozen=True, slots=True)
class _NixlEplbSourceKey:
    generation: int
    layer: int
    expert: int
    source: int
    tensor_group: int = 0


@dataclass(frozen=True, slots=True)
class _NixlEplbTransferKey:
    generation: int
    layer: int
    expert: int
    source: int
    reader: int
    tensor_group: int = 0

    @property
    def source_key(self) -> _NixlEplbSourceKey:
        return _NixlEplbSourceKey(
            generation=self.generation,
            layer=self.layer,
            expert=self.expert,
            source=self.source,
            tensor_group=self.tensor_group,
        )


@dataclass(frozen=True, slots=True)
class _NixlEplbNotification:
    kind: _NixlEplbNotificationKind
    key: _NixlEplbTransferKey

    def encode(self) -> bytes:
        values = {
            "generation": (self.key.generation, (1 << 64) - 1),
            "layer": (self.key.layer, (1 << 32) - 1),
            "expert": (self.key.expert, (1 << 32) - 1),
            "source": (self.key.source, (1 << 32) - 1),
            "reader": (self.key.reader, (1 << 32) - 1),
            "tensor_group": (self.key.tensor_group, (1 << 32) - 1),
        }
        for name, (value, maximum) in values.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"NIXL EPLB {name} must be an integer")
            if not 0 <= value <= maximum:
                raise ValueError(
                    f"NIXL EPLB {name} must be between 0 and {maximum}, got {value}"
                )
        if not isinstance(self.kind, _NixlEplbNotificationKind):
            raise ValueError(f"Invalid NIXL EPLB notification kind: {self.kind!r}")
        if self.kind == _NixlEplbNotificationKind.ABORT:
            raise ValueError("ABORT requires the NIXL EPLB abort payload")
        return _NIXL_EPLB_NOTIFICATION_STRUCT.pack(
            _NIXL_EPLB_NOTIFICATION_MAGIC,
            _NIXL_EPLB_NOTIFICATION_VERSION,
            self.kind,
            self.key.generation,
            self.key.layer,
            self.key.expert,
            self.key.source,
            self.key.reader,
            self.key.tensor_group,
        )

    @classmethod
    def decode(cls, payload: bytes) -> "_NixlEplbNotification":
        if len(payload) != _NIXL_EPLB_NOTIFICATION_STRUCT.size:
            raise ValueError(
                "Invalid NIXL EPLB notification length: "
                f"expected {_NIXL_EPLB_NOTIFICATION_STRUCT.size}, got {len(payload)}"
            )
        (
            magic,
            version,
            kind_value,
            generation,
            layer,
            expert,
            source,
            reader,
            tensor_group,
        ) = _NIXL_EPLB_NOTIFICATION_STRUCT.unpack(payload)
        if magic != _NIXL_EPLB_NOTIFICATION_MAGIC:
            raise ValueError(f"Invalid NIXL EPLB notification magic: {magic!r}")
        if version != _NIXL_EPLB_NOTIFICATION_VERSION:
            raise ValueError(
                "Unsupported NIXL EPLB notification version: "
                f"expected {_NIXL_EPLB_NOTIFICATION_VERSION}, got {version}"
            )
        try:
            kind = _NixlEplbNotificationKind(kind_value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid NIXL EPLB notification kind: {kind_value}"
            ) from exc
        if kind == _NixlEplbNotificationKind.ABORT:
            raise ValueError("ABORT requires the NIXL EPLB abort payload")
        return cls(
            kind=kind,
            key=_NixlEplbTransferKey(
                generation=generation,
                layer=layer,
                expert=expert,
                source=source,
                reader=reader,
                tensor_group=tensor_group,
            ),
        )

    def validate_route(self, sender_rank: int, receiver_rank: int) -> None:
        if self.kind == _NixlEplbNotificationKind.READY:
            expected_sender = self.key.source
            expected_receiver = self.key.reader
        elif self.kind == _NixlEplbNotificationKind.READ_DONE:
            expected_sender = self.key.reader
            expected_receiver = self.key.source
        else:
            raise RuntimeError(f"Invalid NIXL EPLB notification kind: {self.kind!r}")
        if sender_rank != expected_sender or receiver_rank != expected_receiver:
            raise RuntimeError(
                "NIXL EPLB notification route mismatch: "
                f"kind={self.kind.name}, sender={sender_rank}, "
                f"expected_sender={expected_sender}, receiver={receiver_rank}, "
                f"expected_receiver={expected_receiver}, key={self.key}"
            )


@dataclass(frozen=True, slots=True)
class _NixlEplbAbortNotification:
    generation: int
    layer: int
    origin: int
    target: int
    reason: _NixlEplbAbortReason

    def encode(self) -> bytes:
        values = {
            "generation": (self.generation, (1 << 64) - 1),
            "layer": (self.layer, (1 << 32) - 1),
            "origin": (self.origin, (1 << 32) - 1),
            "target": (self.target, (1 << 32) - 1),
        }
        for name, (value, maximum) in values.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"NIXL EPLB ABORT {name} must be an integer")
            if not 0 <= value <= maximum:
                raise ValueError(
                    f"NIXL EPLB ABORT {name} must be between 0 and {maximum}, "
                    f"got {value}"
                )
        if not isinstance(self.reason, _NixlEplbAbortReason):
            raise ValueError(f"Invalid NIXL EPLB ABORT reason: {self.reason!r}")
        return _NIXL_EPLB_ABORT_NOTIFICATION_STRUCT.pack(
            _NIXL_EPLB_NOTIFICATION_MAGIC,
            _NIXL_EPLB_NOTIFICATION_VERSION,
            _NixlEplbNotificationKind.ABORT,
            self.generation,
            self.layer,
            self.origin,
            self.target,
            self.reason,
        )

    @classmethod
    def decode(cls, payload: bytes) -> "_NixlEplbAbortNotification":
        if len(payload) != _NIXL_EPLB_ABORT_NOTIFICATION_STRUCT.size:
            raise ValueError(
                "Invalid NIXL EPLB ABORT notification length: "
                f"expected {_NIXL_EPLB_ABORT_NOTIFICATION_STRUCT.size}, "
                f"got {len(payload)}"
            )
        (
            magic,
            version,
            kind_value,
            generation,
            layer,
            origin,
            target,
            reason_value,
        ) = _NIXL_EPLB_ABORT_NOTIFICATION_STRUCT.unpack(payload)
        if magic != _NIXL_EPLB_NOTIFICATION_MAGIC:
            raise ValueError(f"Invalid NIXL EPLB ABORT magic: {magic!r}")
        if version != _NIXL_EPLB_NOTIFICATION_VERSION:
            raise ValueError(
                "Unsupported NIXL EPLB ABORT version: "
                f"expected {_NIXL_EPLB_NOTIFICATION_VERSION}, got {version}"
            )
        if kind_value != _NixlEplbNotificationKind.ABORT:
            raise ValueError(f"Invalid NIXL EPLB ABORT kind: {kind_value}")
        try:
            reason = _NixlEplbAbortReason(reason_value)
        except ValueError as exc:
            raise ValueError(f"Invalid NIXL EPLB ABORT reason: {reason_value}") from exc
        return cls(
            generation=generation,
            layer=layer,
            origin=origin,
            target=target,
            reason=reason,
        )

    def validate_route(self, sender_rank: int, receiver_rank: int) -> None:
        if sender_rank != self.origin or receiver_rank != self.target:
            raise RuntimeError(
                "NIXL EPLB ABORT route mismatch: "
                f"sender={sender_rank}, expected_sender={self.origin}, "
                f"receiver={receiver_rank}, expected_receiver={self.target}, "
                f"generation={self.generation}, layer={self.layer}"
            )


_NixlEplbProtocolNotification = _NixlEplbNotification | _NixlEplbAbortNotification


def _decode_nixl_eplb_notification(payload: bytes) -> _NixlEplbProtocolNotification:
    if len(payload) < _NIXL_EPLB_NOTIFICATION_HEADER_STRUCT.size:
        raise ValueError(
            "Invalid NIXL EPLB notification length: "
            f"minimum {_NIXL_EPLB_NOTIFICATION_HEADER_STRUCT.size}, "
            f"got {len(payload)}"
        )
    magic, version, kind_value = _NIXL_EPLB_NOTIFICATION_HEADER_STRUCT.unpack_from(
        payload
    )
    if magic != _NIXL_EPLB_NOTIFICATION_MAGIC:
        raise ValueError(f"Invalid NIXL EPLB notification magic: {magic!r}")
    if version != _NIXL_EPLB_NOTIFICATION_VERSION:
        raise ValueError(
            "Unsupported NIXL EPLB notification version: "
            f"expected {_NIXL_EPLB_NOTIFICATION_VERSION}, got {version}"
        )
    try:
        kind = _NixlEplbNotificationKind(kind_value)
    except ValueError as exc:
        raise ValueError(f"Invalid NIXL EPLB notification kind: {kind_value}") from exc
    if kind == _NixlEplbNotificationKind.ABORT:
        return _NixlEplbAbortNotification.decode(payload)
    return _NixlEplbNotification.decode(payload)


class _NixlEplbPeerAbortError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _NixlEplbDeadline:
    phase: _NixlEplbDeadlinePhase
    key: _NixlEplbSourceKey | _NixlEplbTransferKey
    expires_at: float


@dataclass(slots=True)
class _NixlEplbExecuteStats:
    """Per-execute timings and counters for NIXL EPLB."""

    sync_protocol_active: bool
    reads_posted: int = 0
    execute_seconds: float = 0.0
    ready_wait_sum_seconds: float = 0.0
    ready_wait_max_seconds: float = 0.0
    read_post_seconds: float = 0.0
    read_progress_seconds: float = 0.0
    read_completion_sum_seconds: float = 0.0
    read_completion_max_seconds: float = 0.0
    read_done_wait_sum_seconds: float = 0.0
    read_done_wait_max_seconds: float = 0.0
    barrier_seconds: float = 0.0
    ready_sent: int = 0
    ready_received: int = 0
    read_done_attached: int = 0
    read_done_received: int = 0
    duplicate_notifications: int = 0
    stale_notifications: int = 0
    notification_poll_calls: int = 0
    abort_sent: int = 0
    abort_received: int = 0
    abort_send_failures: int = 0
    exclusive_ready_wait_seconds: float = 0.0
    exclusive_read_execution_seconds: float = 0.0
    exclusive_read_done_wait_seconds: float = 0.0
    exclusive_protocol_residual_seconds: float = 0.0


@dataclass(slots=True)
class _NixlEplbPendingRead:
    key: _NixlEplbTransferKey
    tensors: list[torch.Tensor]
    queued_at: float


@dataclass(slots=True)
class _NixlEplbActiveRead:
    key: _NixlEplbTransferKey
    local_dlist: int
    remote_dlist: int
    xfer_handle: int
    posted_at: float
    state: str


class _NixlEplbProtocolState:
    """Pure state for the synchronous notification protocol."""

    def __init__(
        self,
        local_rank: int,
        world_size: int,
        timeout_seconds: float = _NIXL_EPLB_DEFAULT_TIMEOUT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0 <= local_rank < world_size:
            raise ValueError(
                f"local_rank must be in [0, {world_size}), got {local_rank}"
            )
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")
        self.local_rank = local_rank
        self.world_size = world_size
        self.timeout_seconds = timeout_seconds
        self._clock = clock
        self.active_generation: int | None = None
        self.completed_generation = -1
        self._expected_readers: dict[_NixlEplbSourceKey, frozenset[int]] = {}
        self._completed_readers: dict[_NixlEplbSourceKey, set[int]] = {}
        self._seen_ready: set[_NixlEplbTransferKey] = set()
        self._seen_aborts: set[tuple[int, int]] = set()
        self._ready_tokens: set[_NixlEplbTransferKey] = set()
        self._deadlines: dict[
            tuple[
                _NixlEplbDeadlinePhase,
                _NixlEplbSourceKey | _NixlEplbTransferKey,
            ],
            _NixlEplbDeadline,
        ] = {}

    def begin_generation(self, generation: int) -> None:
        if self.active_generation is not None:
            raise RuntimeError(
                "NIXL EPLB synchronous generation overlap: "
                f"active={self.active_generation}, requested={generation}"
            )
        if generation <= self.completed_generation:
            raise RuntimeError(
                "NIXL EPLB generation must advance past completed watermark: "
                f"completed={self.completed_generation}, requested={generation}"
            )
        self.active_generation = generation

    def end_generation(self, success: bool) -> None:
        generation = self.active_generation
        if generation is None:
            raise RuntimeError("No active NIXL EPLB synchronous generation")
        if success:
            self.completed_generation = generation
        self.active_generation = None
        self._expected_readers = {
            key: readers
            for key, readers in self._expected_readers.items()
            if key.generation > generation
        }
        self._completed_readers = {
            key: readers
            for key, readers in self._completed_readers.items()
            if key.generation > generation
        }
        self._ready_tokens = {
            key for key in self._ready_tokens if key.generation > generation
        }
        self._seen_ready = {
            key for key in self._seen_ready if key.generation > generation
        }
        self._seen_aborts = {
            abort_key for abort_key in self._seen_aborts if abort_key[0] > generation
        }
        self._deadlines = {
            deadline_key: deadline
            for deadline_key, deadline in self._deadlines.items()
            if deadline.key.generation > generation
        }

    def set_expected_readers(
        self,
        key: _NixlEplbSourceKey,
        readers: Sequence[int],
    ) -> bool:
        self._require_active_generation(key.generation)
        if key.source != self.local_rank:
            raise RuntimeError(
                "NIXL EPLB expected readers must be owned by the local source: "
                f"local_rank={self.local_rank}, key={key}"
            )
        reader_set = frozenset(readers)
        for reader in reader_set:
            self._validate_rank(reader, "reader")
            if reader == key.source:
                raise RuntimeError(
                    f"NIXL EPLB source cannot be its own remote reader: key={key}"
                )
        existing = self._expected_readers.get(key)
        if existing is not None:
            if existing != reader_set:
                raise RuntimeError(
                    "NIXL EPLB expected readers are already frozen: "
                    f"key={key}, expected={sorted(existing)}, "
                    f"requested={sorted(reader_set)}"
                )
            return False
        self._expected_readers[key] = reader_set
        self._completed_readers[key] = set()
        return True

    def expected_readers(self, key: _NixlEplbSourceKey) -> frozenset[int]:
        return self._expected_readers[key]

    def completed_readers(self, key: _NixlEplbSourceKey) -> frozenset[int]:
        return frozenset(self._completed_readers[key])

    def source_complete(self, key: _NixlEplbSourceKey) -> bool:
        return self.expected_readers(key) == self.completed_readers(key)

    def record_notification(
        self,
        notification: _NixlEplbNotification,
        sender_rank: int,
    ) -> _NixlEplbNotificationDisposition:
        self._validate_rank(sender_rank, "sender")
        notification.validate_route(sender_rank, self.local_rank)
        generation = notification.key.generation
        if generation <= self.completed_generation:
            return _NixlEplbNotificationDisposition.STALE
        if self.active_generation is not None and generation < self.active_generation:
            return _NixlEplbNotificationDisposition.STALE
        self._require_active_generation(generation)
        if notification.kind == _NixlEplbNotificationKind.READY:
            if notification.key in self._seen_ready:
                return _NixlEplbNotificationDisposition.DUPLICATE
            self._seen_ready.add(notification.key)
            self._ready_tokens.add(notification.key)
            return _NixlEplbNotificationDisposition.RECORDED

        source_key = notification.key.source_key
        expected = self._expected_readers.get(source_key)
        if expected is None or notification.key.reader not in expected:
            raise RuntimeError(
                "Unknown NIXL EPLB READ_DONE for active generation: "
                f"key={notification.key}, expected={sorted(expected or ())}"
            )
        completed = self._completed_readers[source_key]
        if notification.key.reader in completed:
            return _NixlEplbNotificationDisposition.DUPLICATE
        completed.add(notification.key.reader)
        return _NixlEplbNotificationDisposition.RECORDED

    def record_abort(
        self,
        notification: _NixlEplbAbortNotification,
        sender_rank: int,
    ) -> _NixlEplbNotificationDisposition:
        self._validate_rank(sender_rank, "sender")
        notification.validate_route(sender_rank, self.local_rank)
        generation = notification.generation
        if generation <= self.completed_generation:
            return _NixlEplbNotificationDisposition.STALE
        if self.active_generation is not None and generation < self.active_generation:
            return _NixlEplbNotificationDisposition.STALE
        self._require_active_generation(generation)
        abort_key = (generation, notification.origin)
        if abort_key in self._seen_aborts:
            return _NixlEplbNotificationDisposition.DUPLICATE
        self._seen_aborts.add(abort_key)
        return _NixlEplbNotificationDisposition.RECORDED

    def consume_ready(self, key: _NixlEplbTransferKey) -> bool:
        self._require_active_generation(key.generation)
        if key not in self._ready_tokens:
            return False
        self._ready_tokens.remove(key)
        return True

    def ready_tokens(self) -> frozenset[_NixlEplbTransferKey]:
        return frozenset(self._ready_tokens)

    def arm_deadline(
        self,
        phase: _NixlEplbDeadlinePhase,
        key: _NixlEplbSourceKey | _NixlEplbTransferKey,
    ) -> _NixlEplbDeadline:
        self._require_active_generation(key.generation)
        deadline = _NixlEplbDeadline(
            phase=phase,
            key=key,
            expires_at=self._clock() + self.timeout_seconds,
        )
        self._deadlines[(phase, key)] = deadline
        return deadline

    def clear_deadline(
        self,
        phase: _NixlEplbDeadlinePhase,
        key: _NixlEplbSourceKey | _NixlEplbTransferKey,
    ) -> None:
        self._deadlines.pop((phase, key), None)

    def expired_deadlines(self) -> tuple[_NixlEplbDeadline, ...]:
        now = self._clock()
        return tuple(
            deadline
            for deadline in self._deadlines.values()
            if now >= deadline.expires_at
        )

    def _require_active_generation(self, generation: int) -> None:
        if generation != self.active_generation:
            raise RuntimeError(
                "NIXL EPLB notification generation mismatch: "
                f"active={self.active_generation}, received={generation}"
            )

    def _validate_rank(self, rank: int, role: str) -> None:
        if not 0 <= rank < self.world_size:
            raise RuntimeError(
                f"NIXL EPLB {role} rank is out of range: "
                f"rank={rank}, world_size={self.world_size}"
            )


def has_nixl() -> bool:
    """Whether the optional NIXL / RIXL package is available."""
    return nixl_utils.NixlWrapper is not None


class EplbCommunicator(ABC):
    """Abstract EPLB communicator for expert weight transfers."""

    @abstractmethod
    def add_send(
        self,
        tensors: list[torch.Tensor],
        dst_rank: int,
        expert_id: int,
    ) -> None:
        pass

    @abstractmethod
    def add_recv(
        self,
        tensors: list[torch.Tensor],
        src_rank: int,
        expert_id: int,
    ) -> None:
        pass

    @abstractmethod
    def execute(self) -> None:
        """Complete all enqueued transfers.

        Some backends perform communication here. NIXL may issue READs
        eagerly in add_recv and progresses any remaining work here.
        On return, all data is available in the destination buffers.
        """

    def set_transfer_context(
        self,
        old_indices: np.ndarray,
        layer_idx: int,
        perf_context: EplbPerfContext | None = None,
    ) -> None:
        """Pre-set layer context before add_recv calls.

        Backends that need layer-level transfer metadata should override this
        method and call ``super().set_transfer_context``.
        """
        self._perf_context = perf_context

    def next_rearrangement_id(self) -> int:
        rearrangement_id = getattr(self, "_next_perf_rearrangement_id", 0)
        self._next_perf_rearrangement_id = rearrangement_id + 1
        return rearrangement_id

    def next_generation_id(self) -> int:
        generation_id = getattr(self, "_next_perf_generation_id", 0)
        self._next_perf_generation_id = generation_id + 1
        return generation_id

    def _set_last_backend_perf(self, perf: EplbBackendPerf) -> None:
        self._last_backend_perf = perf

    def take_last_backend_perf(self) -> EplbBackendPerf:
        perf = getattr(self, "_last_backend_perf", EplbBackendPerf())
        self._last_backend_perf = EplbBackendPerf()
        return perf

    @property
    def perf_label(self) -> str:
        return self.__class__.__name__

    @property
    def needs_profile_buffer_reservation(self) -> bool:
        """Whether the profile path must run a dummy collective operation to reserve
        communication buffers."""
        return True

    def set_stream(self, cuda_stream: torch.cuda.Stream | None) -> None:
        self._cuda_stream = cuda_stream

    def _log_initialized(self) -> None:
        if is_local_first_rank():
            logger.info("Initialized EPLB communicator: %s.", self.__class__.__name__)


class TorchDistNcclEplbCommunicator(EplbCommunicator):
    """EPLB communicator backed by torch.distributed isend/irecv."""

    def __init__(
        self,
        ep_group: ProcessGroup,
        cuda_stream: torch.cuda.Stream | None = None,
    ) -> None:
        self._ep_group = ep_group
        self._cuda_stream = cuda_stream
        self._p2p_ops: list[P2POp] = []
        self._log_initialized()

    def add_send(
        self,
        tensors: list[torch.Tensor],
        dst_rank: int,
        expert_id: int,  # unused by this backend
    ) -> None:
        for tensor in tensors:
            self._p2p_ops.append(
                P2POp(
                    torch.distributed.isend,
                    tensor,
                    dst_rank,
                    self._ep_group,
                )
            )

    def add_recv(
        self,
        tensors: list[torch.Tensor],
        src_rank: int,
        expert_id: int,  # unused by this backend
    ) -> None:
        for tensor in tensors:
            self._p2p_ops.append(
                P2POp(
                    torch.distributed.irecv,
                    tensor,
                    src_rank,
                    self._ep_group,
                )
            )

    def execute(self) -> None:
        if not self._p2p_ops:
            return
        try:
            with torch.cuda.stream(self._cuda_stream):
                reqs = batch_isend_irecv(self._p2p_ops)
                for req in reqs:
                    req.wait()
        finally:
            self._p2p_ops.clear()


class TorchDistGlooStagedEplbCommunicator(EplbCommunicator):
    """EPLB communicator using gloo P2P with CPU staging."""

    def __init__(
        self,
        cpu_group: ProcessGroup,
        cuda_stream: torch.cuda.Stream | None = None,
    ) -> None:
        self._cpu_group = cpu_group
        self._cuda_stream = cuda_stream
        self._ops: list[tuple[str, torch.Tensor, int]] = []
        self._log_initialized()

    def add_send(
        self,
        tensors: list[torch.Tensor],
        dst_rank: int,
        expert_id: int,  # unused by this backend
    ) -> None:
        for tensor in tensors:
            self._ops.append(("send", tensor, dst_rank))

    def add_recv(
        self,
        tensors: list[torch.Tensor],
        src_rank: int,
        expert_id: int,  # unused by this backend
    ) -> None:
        for tensor in tensors:
            self._ops.append(("recv", tensor, src_rank))

    def execute(self) -> None:
        if not self._ops:
            return

        p2p_ops: list[P2POp] = []
        recv_staging: list[tuple[torch.Tensor, torch.Tensor]] = []

        def build_ops() -> None:
            for op, tensor, peer_rank in self._ops:
                if op == "send":
                    cpu_tensor = tensor.to(device="cpu", non_blocking=True)
                    p2p_ops.append(
                        P2POp(
                            torch.distributed.isend,
                            cpu_tensor,
                            peer_rank,
                            self._cpu_group,
                        )
                    )
                    continue
                cpu_tensor = torch.empty_like(tensor, device="cpu")
                p2p_ops.append(
                    P2POp(
                        torch.distributed.irecv,
                        cpu_tensor,
                        peer_rank,
                        self._cpu_group,
                    )
                )
                recv_staging.append((tensor, cpu_tensor))

        try:
            with torch.cuda.stream(self._cuda_stream):
                build_ops()
        finally:
            self._ops.clear()

        # Wait for all D2H copies to finish
        # before issuing gloo batch_isend_irecv operations.
        if self._cuda_stream is not None:
            self._cuda_stream.synchronize()
        else:
            torch.cuda.current_stream().synchronize()

        reqs = batch_isend_irecv(p2p_ops)
        for req in reqs:
            req.wait()

        if not recv_staging:
            return
        with torch.cuda.stream(self._cuda_stream):
            for dst_tensor, cpu_tensor in recv_staging:
                dst_tensor.copy_(cpu_tensor, non_blocking=True)


class NixlEplbCommunicator(EplbCommunicator):
    """EPLB communicator backed by NIXL READ transfers."""

    def __init__(
        self,
        cpu_group: ProcessGroup,
        all_expert_weights: Sequence[Sequence[torch.Tensor]],
        expert_buffer: Sequence[torch.Tensor],
        defer_remote_setup: bool = False,
        enable_sync_protocol: bool = False,
    ) -> None:
        """Create a NIXL-backed EPLB communicator.

        Args:
            cpu_group: CPU process group for metadata exchange.
            all_expert_weights: Expert weight tensors for all MoE layers.
            expert_buffer: Pre-allocated receive buffer tensors.
            defer_remote_setup: If True, postpone the collective
                all-gather of NIXL agent metadata until the first
                ``set_transfer_context`` call.  Required for elastic EP
                where ranks join asynchronously and cannot participate
                in collectives at construction time.
            enable_sync_protocol: Enable the synchronous notification
                protocol. Ignored when remote setup is deferred for elastic EP.
        """
        assert all_expert_weights, (
            "NixlEplbCommunicator requires non-empty all_expert_weights."
        )
        assert expert_buffer, "NixlEplbCommunicator requires non-empty expert_buffer."
        nixl_wrapper_cls = nixl_utils.NixlWrapper
        if nixl_wrapper_cls is None:
            raise RuntimeError("NIXL/ RIXL is unavailable.")

        self._cpu_group = cpu_group
        self._world_size = cpu_group.size()
        self._rank = cpu_group.rank()

        self._all_expert_weights = all_expert_weights
        self._expert_buffer = expert_buffer
        self._num_local_experts: int = all_expert_weights[0][0].shape[0]
        self._device = all_expert_weights[0][0].device

        for layer_tensors in all_expert_weights:
            for tensor in layer_tensors:
                assert is_weak_contiguous(tensor), (
                    "Expert weight tensors must be contiguous in memory"
                )
                assert tensor.device == self._device, (
                    "All local EPLB tensors are expected to be on the same "
                    f"device: expected={self._device}, got={tensor.device}"
                )
        for tensor in expert_buffer:
            assert is_weak_contiguous(tensor), (
                "expert_buffer tensors must be contiguous in memory"
            )

        # Legacy READ transfers used by asynchronous and elastic EPLB.
        self._xfer_entries: list[tuple[int, int, int]] = []
        self._last_execute_stats: _NixlEplbExecuteStats | None = None
        # Per-rank expert_id -> physical row; set by set_transfer_context.
        self._expert_to_src_row: list[dict[int, int]] | None = None
        self._layer_idx: int | None = None

        nixl_agent_config = nixl_utils.nixl_agent_config
        config = (
            nixl_agent_config(capture_telemetry=False)
            if nixl_agent_config is not None
            else None
        )
        self._nixl_wrapper = nixl_wrapper_cls(self._make_agent_name(), config)
        self._nixl_memory_type = "VRAM"
        # NIXL registration handles; deregistered in __del__.
        self._registered_descs: list[object] = []
        self._remote_agents: dict[int, str | bytes] = {}
        self._remote_agent_ranks: dict[str, int] = {}
        # peer -> (layer, tensor) -> (base_ptr, bytes_per_expert, dev_id).
        self._remote_send_meta: dict[
            int, dict[tuple[int, int], tuple[int, int, int]]
        ] = {}

        self._sync_protocol_active = enable_sync_protocol and not defer_remote_setup
        self._sync_protocol_state = _NixlEplbProtocolState(
            local_rank=self._rank,
            world_size=self._world_size,
        )
        self._next_sync_generation = 0
        self._sync_stats: _NixlEplbExecuteStats | None = None
        self._pending_reads: dict[_NixlEplbTransferKey, _NixlEplbPendingRead] = {}
        self._active_reads: dict[_NixlEplbTransferKey, _NixlEplbActiveRead] = {}
        self._recv_keys: set[_NixlEplbTransferKey] = set()
        self._source_readers: dict[_NixlEplbSourceKey, set[int]] = {}
        self._source_expectations_frozen = False
        self._ready_sent_at: dict[_NixlEplbTransferKey, float] = {}
        self._deferred_notifications: dict[
            int, list[tuple[_NixlEplbProtocolNotification, int]]
        ] = {}
        self._protocol_failed = False
        self._protocol_failure: str | None = None
        self._exclusive_phase_timer: _NixlEplbExclusivePhaseTimer | None = None

        self._cuda_device_id = int(self._device.index or 0)
        self._remote_state_initialized = False
        self._init_step("buffers", self._init_registered_buffers)
        if defer_remote_setup:
            logger.info_once("NIXL EPLB: deferring remote agent setup (elastic EP).")
        else:
            self._init_remote_state()
        self._log_initialized()

    def _init_remote_state(self) -> None:
        """Exchange NIXL agent metadata and RDMA pointer info with all peers.

        This is a collective operation (uses ``all_gather_object`` twice).
        Under elastic EP the call is deferred to the first
        ``set_transfer_context`` invocation, where all ranks are
        guaranteed to be synchronized.
        """
        self._init_step("agents", self._init_remote_agents)
        self._init_step("send meta", self._exchange_remote_send_meta)
        self._remote_state_initialized = True

    def _ensure_remote_state(self) -> None:
        if not self._remote_state_initialized:
            self._init_remote_state()

    @property
    def needs_profile_buffer_reservation(self) -> bool:
        return False

    @property
    def perf_label(self) -> str:
        return "candidate_nixl" if self._sync_protocol_active else "baseline_nixl"

    def next_generation_id(self) -> int:
        if self._sync_protocol_active:
            return self._next_sync_generation
        return super().next_generation_id()

    @staticmethod
    def _init_step(name: str, fn: object, *args: object, **kwargs: object) -> None:
        try:
            fn(*args, **kwargs)  # type: ignore[operator]
        except Exception as exc:
            raise RuntimeError(f"NIXL EPLB init failed: {name}") from exc

    def _make_agent_name(self) -> str:
        """Build a deployment-unique nixl agent name."""
        pp_size = get_pp_group().world_size
        pp_suffix = f"-pp{get_pp_group().rank_in_group}" if pp_size > 1 else ""
        uid = uuid.uuid4().hex[:8]
        return f"eplb-{self._rank}{pp_suffix}-{uid}"

    def set_stream(self, cuda_stream: torch.cuda.Stream | None) -> None:
        pass

    def _raise_if_sync_protocol_failed(self) -> None:
        if self._protocol_failed:
            raise RuntimeError(
                "NIXL EPLB synchronous communicator failed and cannot be reused; "
                "recreate the communicator or worker. "
                f"previous_failure={self._protocol_failure}"
            )

    @staticmethod
    def _mark_abort_reason(
        exc: Exception,
        reason: _NixlEplbAbortReason,
    ) -> None:
        with contextlib.suppress(Exception):
            exc._nixl_eplb_abort_reason = reason  # type: ignore[attr-defined]

    @staticmethod
    def _abort_reason_for_exception(exc: Exception) -> _NixlEplbAbortReason:
        reason = getattr(
            exc,
            "_nixl_eplb_abort_reason",
            _NixlEplbAbortReason.PROTOCOL_ERROR,
        )
        if isinstance(reason, _NixlEplbAbortReason):
            return reason
        return _NixlEplbAbortReason.PROTOCOL_ERROR

    def _broadcast_abort(self, reason: _NixlEplbAbortReason) -> None:
        generation = self._sync_protocol_state.active_generation
        layer = self._layer_idx
        if generation is None or layer is None:
            return
        for peer, agent_name in self._remote_agents.items():
            notification = _NixlEplbAbortNotification(
                generation=generation,
                layer=layer,
                origin=self._rank,
                target=peer,
                reason=reason,
            )
            try:
                self._nixl_wrapper.send_notif(
                    agent_name,
                    notif_msg=notification.encode(),
                )
                if self._sync_stats is not None:
                    self._sync_stats.abort_sent += 1
            except Exception as exc:
                if self._sync_stats is not None:
                    self._sync_stats.abort_send_failures += 1
                logger.warning(
                    "NIXL EPLB failed to send ABORT: rank=%d peer=%d "
                    "generation=%d layer=%d reason=%s error=%r",
                    self._rank,
                    peer,
                    generation,
                    layer,
                    reason.name,
                    exc,
                )

    def _fail_sync_protocol(self, exc: Exception, *, broadcast: bool) -> None:
        if self._protocol_failed:
            return
        if broadcast:
            self._broadcast_abort(self._abort_reason_for_exception(exc))
        self._protocol_failed = True
        self._protocol_failure = f"{type(exc).__name__}: {exc}"
        if self._sync_stats is not None:
            self._last_execute_stats = self._sync_stats
        if self._sync_protocol_state.active_generation is not None:
            with contextlib.suppress(Exception):
                self._sync_protocol_state.end_generation(success=False)
        self._clear_transfer_state()

    def add_send(
        self,
        tensors: list[torch.Tensor],
        dst_rank: int,
        expert_id: int,
    ) -> None:
        if not self._sync_protocol_active:
            return
        self._raise_if_sync_protocol_failed()
        try:
            key = self._make_transfer_key(
                expert_id=expert_id,
                source=self._rank,
                reader=dst_rank,
            )
            if self._source_expectations_frozen:
                raise RuntimeError(
                    "NIXL EPLB add_send() must precede add_recv() and execute()"
                )
            if key in self._ready_sent_at:
                raise RuntimeError(f"Duplicate NIXL EPLB add_send(): key={key}")

            readers = self._source_readers.setdefault(key.source_key, set())
            readers.add(dst_rank)
            notification = _NixlEplbNotification(
                _NixlEplbNotificationKind.READY,
                key,
            )
            sent_at = time.perf_counter()
            try:
                self._nixl_wrapper.send_notif(
                    self._remote_agents[dst_rank],
                    notif_msg=notification.encode(),
                )
            except Exception:
                readers.remove(dst_rank)
                if not readers:
                    del self._source_readers[key.source_key]
                raise
            self._ready_sent_at[key] = sent_at
            assert self._sync_stats is not None
            self._sync_stats.ready_sent += 1
        except Exception as exc:
            self._fail_sync_protocol(exc, broadcast=True)
            raise

    def set_transfer_context(
        self,
        old_indices: np.ndarray,
        layer_idx: int,
        perf_context: EplbPerfContext | None = None,
    ) -> None:
        super().set_transfer_context(old_indices, layer_idx, perf_context)
        if self._sync_protocol_active:
            self._raise_if_sync_protocol_failed()
        self._ensure_remote_state()
        pending_count = (
            len(self._xfer_entries) + len(self._pending_reads) + len(self._active_reads)
        )
        generation_active = self._sync_protocol_state.active_generation is not None
        assert not pending_count and not generation_active, (
            f"set_transfer_context() called with {pending_count} pending transfers "
            f"from layer {self._layer_idx}; execute() was not called after the "
            "previous transfer setup"
        )
        self._read_payloads = []
        self._layer_idx = layer_idx
        n = self._num_local_experts
        rank_experts = old_indices[: self._world_size * n].reshape(self._world_size, n)
        self._expert_to_src_row = [
            {int(eid): i for i, eid in enumerate(row) if eid != -1}
            for row in rank_experts
        ]
        if self._sync_protocol_active:
            generation = self._next_sync_generation
            self._next_sync_generation += 1
            if perf_context is not None:
                assert generation == perf_context.generation_id, (
                    "NIXL protocol and performance generation IDs diverged: "
                    f"protocol={generation}, perf={perf_context.generation_id}"
                )
            self._sync_protocol_state.begin_generation(generation)
            self._sync_stats = _NixlEplbExecuteStats(sync_protocol_active=True)
            self._exclusive_phase_timer = _NixlEplbExclusivePhaseTimer()
            self._source_expectations_frozen = False

    def add_recv(
        self,
        tensors: list[torch.Tensor],
        src_rank: int,
        expert_id: int,
    ) -> None:
        if self._sync_protocol_active:
            self._raise_if_sync_protocol_failed()
        assert self._expert_to_src_row is not None and self._layer_idx is not None, (
            "set_transfer_context() must be called before add_recv()"
        )
        if not self._sync_protocol_active:
            self._post_legacy_read(tensors, src_rank, expert_id)
            return

        try:
            key = self._make_transfer_key(
                expert_id=expert_id,
                source=src_rank,
                reader=self._rank,
            )
            if key in self._recv_keys:
                raise RuntimeError(f"Duplicate NIXL EPLB add_recv(): key={key}")
            self._recv_keys.add(key)
            request = _NixlEplbPendingRead(
                key=key,
                tensors=tensors,
                queued_at=time.perf_counter(),
            )

            self._freeze_source_expectations()
            self._progress_notifications_once()
            if self._sync_protocol_state.consume_ready(key):
                self._record_ready_wait(request)
                self._post_sync_read(request)
                return

            self._pending_reads[key] = request
            self._sync_protocol_state.arm_deadline(
                _NixlEplbDeadlinePhase.READY,
                key,
            )
            self._update_exclusive_phase()
        except _NixlEplbPeerAbortError as exc:
            self._fail_sync_protocol(exc, broadcast=False)
            raise
        except Exception as exc:
            self._fail_sync_protocol(exc, broadcast=True)
            raise

    def _make_transfer_key(
        self,
        *,
        expert_id: int,
        source: int,
        reader: int,
    ) -> _NixlEplbTransferKey:
        generation = self._sync_protocol_state.active_generation
        if generation is None or self._layer_idx is None:
            raise RuntimeError(
                "set_transfer_context() must be called before NIXL EPLB transfers"
            )
        return _NixlEplbTransferKey(
            generation=generation,
            layer=self._layer_idx,
            expert=expert_id,
            source=source,
            reader=reader,
        )

    def _build_read_descs(
        self,
        tensors: list[torch.Tensor],
        src_rank: int,
        expert_id: int,
    ) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
        assert self._expert_to_src_row is not None and self._layer_idx is not None
        src_row = self._expert_to_src_row[src_rank][expert_id]
        layer_idx = self._layer_idx

        local_descs: list[tuple[int, int, int]] = []
        remote_descs: list[tuple[int, int, int]] = []
        for t_idx, t in enumerate(tensors):
            send_base, send_stride, remote_dev = self._remote_send_meta[src_rank][
                (layer_idx, t_idx)
            ]
            assert t.nbytes == send_stride, (
                f"tensor {t_idx} size {t.nbytes} != remote stride {send_stride}"
            )
            local_descs.append(
                (
                    t.data_ptr(),
                    t.nbytes,
                    self._cuda_device_id,
                )
            )
            remote_descs.append(
                (
                    send_base + src_row * send_stride,
                    send_stride,
                    remote_dev,
                )
            )
        return local_descs, remote_descs

    def _post_legacy_read(
        self,
        tensors: list[torch.Tensor],
        src_rank: int,
        expert_id: int,
    ) -> None:
        local_descs, remote_descs = self._build_read_descs(
            tensors,
            src_rank,
            expert_id,
        )
        self._record_read_payload(src_rank, expert_id, local_descs)
        local_h, remote_h, xfer_h = self._create_peer_xfer(
            src_rank, local_descs, remote_descs
        )
        self._nixl_wrapper.transfer(xfer_h)
        self._xfer_entries.append((local_h, remote_h, xfer_h))

    def _init_remote_agents(self) -> None:
        local_metadata = self._nixl_wrapper.get_agent_metadata()
        gathered_metadata: list[bytes | None] = [None] * self._world_size
        torch.distributed.all_gather_object(
            gathered_metadata, local_metadata, group=self._cpu_group
        )
        for peer in range(self._world_size):
            if peer == self._rank:
                continue
            peer_metadata = gathered_metadata[peer]
            assert peer_metadata is not None
            agent_handle = self._nixl_wrapper.add_remote_agent(peer_metadata)
            agent_name = _normalize_nixl_agent_name(agent_handle)
            previous_rank = self._remote_agent_ranks.get(agent_name)
            if previous_rank is not None:
                raise RuntimeError(
                    "NIXL EPLB remote agent name is not unique: "
                    f"agent={agent_name!r}, ranks={previous_rank},{peer}"
                )
            self._remote_agents[peer] = agent_handle
            self._remote_agent_ranks[agent_name] = peer

    def _init_registered_buffers(self) -> None:
        all_tensors: list[torch.Tensor] = []
        for layer_tensors in self._all_expert_weights:
            all_tensors.extend(layer_tensors)
        all_tensors.extend(self._expert_buffer)

        descs = self._nixl_wrapper.get_reg_descs(all_tensors)
        self._nixl_wrapper.register_memory(descs)
        self._registered_descs.append(descs)

    def _exchange_remote_send_meta(self) -> None:
        """Exchange per-layer per-tensor metadata so receivers can compute
        remote RDMA addresses at transfer time."""
        local_meta: dict[tuple[int, int], tuple[int, int, int]] = {}
        for layer_idx, layer_tensors in enumerate(self._all_expert_weights):
            for t_idx, t in enumerate(layer_tensors):
                nbytes_per_expert = t.nbytes // self._num_local_experts
                local_meta[(layer_idx, t_idx)] = (
                    t.data_ptr(),
                    nbytes_per_expert,
                    self._cuda_device_id,
                )

        # Per-rank map: (layer_idx, tensor_idx) -> (base_ptr, bytes_per_expert, dev_id).
        # add_recv uses base_ptr + src_row * bytes_per_expert to compute
        # the remote RDMA address for each expert.
        gathered_meta: list[dict[tuple[int, int], tuple[int, int, int]] | None] = [
            None
        ] * self._world_size
        torch.distributed.all_gather_object(
            gathered_meta, local_meta, group=self._cpu_group
        )

        local_keys = set(local_meta.keys())
        for peer in self._remote_agents:
            peer_meta = gathered_meta[peer]
            assert peer_meta is not None
            peer_keys = set(peer_meta.keys())
            if peer_keys != local_keys:
                raise RuntimeError(
                    f"NIXL EPLB metadata key mismatch with rank {peer}: "
                    f"local={sorted(local_keys)}, peer={sorted(peer_keys)}"
                )
            for key in local_keys:
                _, local_stride, _ = local_meta[key]
                _, peer_stride, _ = peer_meta[key]
                if local_stride != peer_stride:
                    raise RuntimeError(
                        f"NIXL EPLB nbytes_per_expert mismatch for {key} "
                        f"with rank {peer}: "
                        f"local={local_stride}, peer={peer_stride}"
                    )
            self._remote_send_meta[peer] = peer_meta

    def _freeze_source_expectations(self) -> None:
        if self._source_expectations_frozen:
            return
        for key, readers in self._source_readers.items():
            self._sync_protocol_state.set_expected_readers(key, list(readers))
            if readers:
                self._sync_protocol_state.arm_deadline(
                    _NixlEplbDeadlinePhase.READ_DONE,
                    key,
                )
        self._source_expectations_frozen = True
        self._update_exclusive_phase()

    def _progress_notifications_once(self) -> bool:
        assert self._source_expectations_frozen
        assert self._sync_stats is not None
        made_progress = self._process_deferred_notifications()
        self._sync_stats.notification_poll_calls += 1
        notifications = self._nixl_wrapper.get_new_notifs()
        for sender_name, payloads in notifications.items():
            normalized_name = _normalize_nixl_agent_name(sender_name)
            sender_rank = self._remote_agent_ranks.get(normalized_name)
            if sender_rank is None:
                raise RuntimeError(
                    "NIXL EPLB notification from unknown agent: "
                    f"agent={normalized_name!r}"
                )
            for payload in payloads:
                notification = _decode_nixl_eplb_notification(payload)
                made_progress = (
                    self._record_or_defer_notification(notification, sender_rank)
                    or made_progress
                )
        return self._post_ready_reads() or made_progress

    def _process_deferred_notifications(self) -> bool:
        generation = self._sync_protocol_state.active_generation
        assert generation is not None
        deferred = self._deferred_notifications.pop(generation, ())
        for notification, sender_rank in deferred:
            self._record_notification(notification, sender_rank)
        return bool(deferred)

    def _record_or_defer_notification(
        self,
        notification: _NixlEplbProtocolNotification,
        sender_rank: int,
    ) -> bool:
        notification.validate_route(sender_rank, self._rank)
        active_generation = self._sync_protocol_state.active_generation
        assert active_generation is not None
        generation = (
            notification.generation
            if isinstance(notification, _NixlEplbAbortNotification)
            else notification.key.generation
        )
        if generation > active_generation:
            self._deferred_notifications.setdefault(
                generation,
                [],
            ).append((notification, sender_rank))
            return True
        self._record_notification(notification, sender_rank)
        return True

    def _record_notification(
        self,
        notification: _NixlEplbProtocolNotification,
        sender_rank: int,
    ) -> None:
        assert self._sync_stats is not None
        if isinstance(notification, _NixlEplbAbortNotification):
            disposition = self._sync_protocol_state.record_abort(
                notification,
                sender_rank,
            )
        else:
            disposition = self._sync_protocol_state.record_notification(
                notification,
                sender_rank,
            )
        if disposition == _NixlEplbNotificationDisposition.DUPLICATE:
            self._sync_stats.duplicate_notifications += 1
            return
        if disposition == _NixlEplbNotificationDisposition.STALE:
            self._sync_stats.stale_notifications += 1
            return
        if isinstance(notification, _NixlEplbAbortNotification):
            self._sync_stats.abort_received += 1
            raise _NixlEplbPeerAbortError(
                "NIXL EPLB synchronous generation aborted by peer: "
                f"generation={notification.generation}, layer={notification.layer}, "
                f"origin={notification.origin}, reason={notification.reason.name}"
            )
        if notification.kind == _NixlEplbNotificationKind.READY:
            self._sync_stats.ready_received += 1
            self._update_exclusive_phase()
            return

        self._sync_stats.read_done_received += 1
        started_at = self._ready_sent_at.pop(notification.key, None)
        if started_at is not None:
            elapsed = time.perf_counter() - started_at
            self._sync_stats.read_done_wait_sum_seconds += elapsed
            self._sync_stats.read_done_wait_max_seconds = max(
                self._sync_stats.read_done_wait_max_seconds,
                elapsed,
            )
        source_key = notification.key.source_key
        if self._sync_protocol_state.source_complete(source_key):
            self._sync_protocol_state.clear_deadline(
                _NixlEplbDeadlinePhase.READ_DONE,
                source_key,
            )
        self._update_exclusive_phase()

    def _post_ready_reads(self) -> bool:
        made_progress = False
        for key, request in list(self._pending_reads.items()):
            if not self._sync_protocol_state.consume_ready(key):
                continue
            del self._pending_reads[key]
            self._sync_protocol_state.clear_deadline(
                _NixlEplbDeadlinePhase.READY,
                key,
            )
            self._record_ready_wait(request)
            self._post_sync_read(request)
            made_progress = True
        self._update_exclusive_phase()
        return made_progress

    def _record_ready_wait(self, request: _NixlEplbPendingRead) -> None:
        assert self._sync_stats is not None
        elapsed = time.perf_counter() - request.queued_at
        self._sync_stats.ready_wait_sum_seconds += elapsed
        self._sync_stats.ready_wait_max_seconds = max(
            self._sync_stats.ready_wait_max_seconds,
            elapsed,
        )

    def _post_sync_read(self, request: _NixlEplbPendingRead) -> None:
        assert self._sync_stats is not None
        post_started = time.perf_counter()
        local_h: int | None = None
        remote_h: int | None = None
        xfer_h: int | None = None
        try:
            local_descs, remote_descs = self._build_read_descs(
                request.tensors,
                request.key.source,
                request.key.expert,
            )
            self._record_read_payload(
                request.key.source,
                request.key.expert,
                local_descs,
            )
            read_done = _NixlEplbNotification(
                _NixlEplbNotificationKind.READ_DONE,
                request.key,
            )
            local_h, remote_h, xfer_h = self._create_peer_xfer(
                request.key.source,
                local_descs,
                remote_descs,
                notif_msg=read_done.encode(),
            )
            read_started = time.perf_counter()
            state = self._nixl_wrapper.transfer(xfer_h)
            if state not in ("DONE", "PROC"):
                exc = RuntimeError(f"NIXL transfer failed with state={state}")
                self._mark_abort_reason(exc, _NixlEplbAbortReason.READ_FAILURE)
                raise exc
            self._sync_protocol_state.arm_deadline(
                _NixlEplbDeadlinePhase.READ,
                request.key,
            )
            self._active_reads[request.key] = _NixlEplbActiveRead(
                key=request.key,
                local_dlist=local_h,
                remote_dlist=remote_h,
                xfer_handle=xfer_h,
                posted_at=read_started,
                state=state,
            )
            self._sync_stats.reads_posted += 1
            self._sync_stats.read_done_attached += 1
            self._update_exclusive_phase()
        except Exception as exc:
            self._mark_abort_reason(exc, _NixlEplbAbortReason.READ_FAILURE)
            if xfer_h is not None:
                with contextlib.suppress(Exception):
                    self._nixl_wrapper.release_xfer_handle(xfer_h)
            if local_h is not None:
                with contextlib.suppress(Exception):
                    self._nixl_wrapper.release_dlist_handle(local_h)
            if remote_h is not None:
                with contextlib.suppress(Exception):
                    self._nixl_wrapper.release_dlist_handle(remote_h)
            raise
        finally:
            self._sync_stats.read_post_seconds += time.perf_counter() - post_started

    def _record_read_payload(
        self,
        src_rank: int,
        expert_id: int,
        local_descs: list[tuple[int, int, int]],
    ) -> None:
        read_payloads = getattr(self, "_read_payloads", None)
        if read_payloads is None:
            read_payloads = self._read_payloads = []
        read_payloads.append(
            (src_rank, expert_id, sum(desc[1] for desc in local_descs))
        )

    def _progress_active_reads(self) -> bool:
        assert self._sync_stats is not None
        made_progress = False
        for key, entry in list(self._active_reads.items()):
            state = entry.state
            if state == "PROC":
                progress_started = time.perf_counter()
                try:
                    state = self._nixl_wrapper.check_xfer_state(entry.xfer_handle)
                except Exception as exc:
                    self._mark_abort_reason(
                        exc,
                        _NixlEplbAbortReason.READ_FAILURE,
                    )
                    raise
                finally:
                    self._sync_stats.read_progress_seconds += (
                        time.perf_counter() - progress_started
                    )
                entry.state = state
            if state == "PROC":
                continue
            if state != "DONE":
                exc = RuntimeError(f"NIXL transfer failed with state={state}")
                self._mark_abort_reason(exc, _NixlEplbAbortReason.READ_FAILURE)
                raise exc

            elapsed = time.perf_counter() - entry.posted_at
            self._sync_stats.read_completion_sum_seconds += elapsed
            self._sync_stats.read_completion_max_seconds = max(
                self._sync_stats.read_completion_max_seconds,
                elapsed,
            )
            self._sync_protocol_state.clear_deadline(
                _NixlEplbDeadlinePhase.READ,
                key,
            )
            self._release_active_read(entry)
            del self._active_reads[key]
            made_progress = True
        self._update_exclusive_phase()
        return made_progress

    def _current_exclusive_phase(self) -> _NixlEplbPerfPhase:
        if self._active_reads:
            return _NixlEplbPerfPhase.READ_EXECUTION
        if self._pending_reads:
            return _NixlEplbPerfPhase.READY_WAIT
        if self._source_expectations_frozen and any(
            not self._sync_protocol_state.source_complete(key)
            for key in self._source_readers
        ):
            return _NixlEplbPerfPhase.READ_DONE_WAIT
        return _NixlEplbPerfPhase.PROTOCOL_RESIDUAL

    def _update_exclusive_phase(self) -> None:
        timer = getattr(self, "_exclusive_phase_timer", None)
        if timer is not None:
            timer.transition(self._current_exclusive_phase())

    def _finish_exclusive_phase_timing(
        self,
        stats: _NixlEplbExecuteStats,
    ) -> None:
        timer = getattr(self, "_exclusive_phase_timer", None)
        if timer is None:
            return
        totals = timer.finish()
        stats.exclusive_ready_wait_seconds = totals[_NixlEplbPerfPhase.READY_WAIT]
        stats.exclusive_read_execution_seconds = totals[
            _NixlEplbPerfPhase.READ_EXECUTION
        ]
        stats.exclusive_read_done_wait_seconds = totals[
            _NixlEplbPerfPhase.READ_DONE_WAIT
        ]
        stats.exclusive_protocol_residual_seconds = totals[
            _NixlEplbPerfPhase.PROTOCOL_RESIDUAL
        ]
        self._exclusive_phase_timer = None

    def _release_active_read(self, entry: _NixlEplbActiveRead) -> None:
        with contextlib.suppress(Exception):
            self._nixl_wrapper.release_xfer_handle(entry.xfer_handle)
        with contextlib.suppress(Exception):
            self._nixl_wrapper.release_dlist_handle(entry.local_dlist)
        with contextlib.suppress(Exception):
            self._nixl_wrapper.release_dlist_handle(entry.remote_dlist)

    def _sync_work_complete(self) -> bool:
        if self._pending_reads or self._active_reads:
            return False
        return all(
            self._sync_protocol_state.source_complete(key)
            for key in self._source_readers
        )

    def _raise_for_unmatched_ready(self) -> None:
        unmatched = self._sync_protocol_state.ready_tokens()
        if unmatched:
            raise RuntimeError(
                "NIXL EPLB received READY without matching add_recv(): "
                f"keys={sorted(unmatched, key=repr)}"
            )

    def _raise_for_expired_deadlines(self) -> None:
        expired = self._sync_protocol_state.expired_deadlines()
        if not expired:
            return
        details = ", ".join(
            f"phase={deadline.phase.value}, key={deadline.key}" for deadline in expired
        )
        reasons = {
            _NixlEplbDeadlinePhase.READY: _NixlEplbAbortReason.READY_TIMEOUT,
            _NixlEplbDeadlinePhase.READ: _NixlEplbAbortReason.READ_TIMEOUT,
            _NixlEplbDeadlinePhase.READ_DONE: (_NixlEplbAbortReason.READ_DONE_TIMEOUT),
        }
        reason = reasons.get(
            expired[0].phase,
            _NixlEplbAbortReason.PROTOCOL_ERROR,
        )
        exc = RuntimeError(f"NIXL EPLB synchronous protocol timed out: {details}")
        self._mark_abort_reason(exc, reason)
        raise exc

    def _wait_for_all_transfers(self, handles: list[int]) -> None:
        pending = set(handles)
        while pending:
            completed: list[int] = []
            for handle in pending:
                state = self._nixl_wrapper.check_xfer_state(handle)
                if state == "DONE":
                    completed.append(handle)
                    continue
                if state != "PROC":
                    raise RuntimeError(f"NIXL transfer failed with state={state}")
            for handle in completed:
                pending.remove(handle)
            if pending:
                time.sleep(0.0005)

    def _create_peer_xfer(
        self,
        src: int,
        local_descs: list[tuple[int, int, int]],
        remote_descs: list[tuple[int, int, int]],
        notif_msg: bytes | None = None,
    ) -> tuple[int, int, int]:
        """Create a batched xfer for multiple descriptors from one peer.

        Each element in *local_descs* / *remote_descs* is an
        ``(address, size, device_id)`` tuple.

        Returns ``(local_dlist, remote_dlist, xfer_handle)``.
        """
        local_desc = self._nixl_wrapper.get_xfer_descs(
            local_descs, self._nixl_memory_type
        )
        local_handle = self._nixl_wrapper.prep_xfer_dlist(
            "NIXL_INIT_AGENT",
            local_desc,
        )

        remote_desc = self._nixl_wrapper.get_xfer_descs(
            remote_descs, self._nixl_memory_type
        )
        remote_handle = self._nixl_wrapper.prep_xfer_dlist(
            self._remote_agents[src],
            remote_desc,
        )

        indices = list(range(len(local_descs)))
        if notif_msg is None:
            xfer_handle = self._nixl_wrapper.make_prepped_xfer(
                "READ",
                local_handle,
                indices,
                remote_handle,
                indices,
            )
        else:
            xfer_handle = self._nixl_wrapper.make_prepped_xfer(
                "READ",
                local_handle,
                indices,
                remote_handle,
                indices,
                notif_msg=notif_msg,
            )
        return (local_handle, remote_handle, xfer_handle)

    def _post_read_barrier(self) -> None:
        """Correctness fence: prevents overwrite-while-remote-read race.

        We avoid ``torch.distributed.monitored_barrier`` because it
        calls ``get_backend(group)`` which fails for stateless groups
        (elastic EP).  An async ``all_reduce`` + ``wait(timeout)``
        works with both regular and stateless groups and provides
        equivalent timeout detection.
        """
        _dummy = torch.zeros(1, dtype=torch.int32)
        work = torch.distributed.all_reduce(
            _dummy, group=self._cpu_group, async_op=True
        )
        work.wait(timeout=timedelta(minutes=5))

    def _clear_transfer_state(self) -> None:
        for local_h, remote_h, xfer_h in self._xfer_entries:
            with contextlib.suppress(Exception):
                self._nixl_wrapper.release_xfer_handle(xfer_h)
            with contextlib.suppress(Exception):
                self._nixl_wrapper.release_dlist_handle(local_h)
            with contextlib.suppress(Exception):
                self._nixl_wrapper.release_dlist_handle(remote_h)
        self._xfer_entries.clear()
        for entry in self._active_reads.values():
            self._release_active_read(entry)
        self._active_reads.clear()
        self._pending_reads.clear()
        self._recv_keys.clear()
        self._source_readers.clear()
        self._ready_sent_at.clear()
        self._source_expectations_frozen = False
        self._sync_stats = None
        self._expert_to_src_row = None
        self._layer_idx = None

    def execute(self) -> None:
        has_transfers = bool(
            self._xfer_entries or self._pending_reads or self._active_reads
        )
        assert self._layer_idx is not None or not has_transfers, (
            "set_transfer_context() must be called before execute() "
            "if any transfers were added"
        )
        if not self._sync_protocol_active:
            execute_started = time.perf_counter()
            wait_seconds = 0.0
            barrier_seconds = 0.0
            try:
                wait_started = time.perf_counter()
                self._wait_for_all_transfers([x[2] for x in self._xfer_entries])
                wait_seconds = time.perf_counter() - wait_started
                barrier_started = time.perf_counter()
                self._post_read_barrier()
                barrier_seconds = time.perf_counter() - barrier_started
            finally:
                self._set_last_backend_perf(
                    EplbBackendPerf(
                        fields={
                            "backend_wall_ms": (time.perf_counter() - execute_started)
                            * 1000,
                            "transfer_wait_ms": wait_seconds * 1000,
                            "barrier_ms": barrier_seconds * 1000,
                        },
                        read_payloads=getattr(self, "_read_payloads", []).copy(),
                    )
                )
                self._clear_transfer_state()
            return

        self._raise_if_sync_protocol_failed()
        if self._layer_idx is None or self._sync_stats is None:
            raise RuntimeError(
                "set_transfer_context() must be called before synchronous "
                "NIXL EPLB execute()"
            )
        stats = self._sync_stats
        generation = self._sync_protocol_state.active_generation
        assert generation is not None
        execute_started = time.perf_counter()
        try:
            self._freeze_source_expectations()
            while True:
                made_progress = self._progress_notifications_once()
                self._raise_for_unmatched_ready()
                made_progress = self._progress_active_reads() or made_progress
                if self._sync_work_complete():
                    break
                self._raise_for_expired_deadlines()
                if not made_progress:
                    time.sleep(0.0005)
            self._sync_protocol_state.end_generation(success=True)
            self._clear_transfer_state()
        except _NixlEplbPeerAbortError as exc:
            self._fail_sync_protocol(exc, broadcast=False)
            raise
        except Exception as exc:
            self._fail_sync_protocol(exc, broadcast=True)
            raise
        finally:
            stats.execute_seconds = time.perf_counter() - execute_started
            self._finish_exclusive_phase_timing(stats)
            self._last_execute_stats = stats
            self._set_last_backend_perf(
                EplbBackendPerf(
                    fields={
                        "backend_wall_ms": stats.execute_seconds * 1000,
                        "ready_wait_sum_ms": (stats.ready_wait_sum_seconds * 1000),
                        "read_completion_sum_ms": (
                            stats.read_completion_sum_seconds * 1000
                        ),
                        "read_done_wait_sum_ms": (
                            stats.read_done_wait_sum_seconds * 1000
                        ),
                        "ready_wait_ms": (stats.exclusive_ready_wait_seconds * 1000),
                        "read_execution_ms": (
                            stats.exclusive_read_execution_seconds * 1000
                        ),
                        "read_done_wait_ms": (
                            stats.exclusive_read_done_wait_seconds * 1000
                        ),
                        "protocol_residual_ms": (
                            stats.exclusive_protocol_residual_seconds * 1000
                        ),
                    },
                    read_payloads=getattr(self, "_read_payloads", []).copy(),
                )
            )
            logger.debug(
                "NIXL EPLB execute stats: rank=%d generation=%d "
                "sync_protocol=%s reads=%d execute_ms=%.3f "
                "ready_wait_sum_ms=%.3f ready_wait_max_ms=%.3f "
                "read_post_ms=%.3f read_progress_ms=%.3f "
                "read_completion_sum_ms=%.3f read_completion_max_ms=%.3f "
                "read_done_wait_sum_ms=%.3f read_done_wait_max_ms=%.3f "
                "barrier_ms=%.3f ready_sent=%d ready_received=%d "
                "read_done_attached=%d read_done_received=%d "
                "duplicate_notifications=%d stale_notifications=%d polls=%d "
                "abort_sent=%d abort_received=%d abort_send_failures=%d",
                self._rank,
                generation,
                stats.sync_protocol_active,
                stats.reads_posted,
                stats.execute_seconds * 1000,
                stats.ready_wait_sum_seconds * 1000,
                stats.ready_wait_max_seconds * 1000,
                stats.read_post_seconds * 1000,
                stats.read_progress_seconds * 1000,
                stats.read_completion_sum_seconds * 1000,
                stats.read_completion_max_seconds * 1000,
                stats.read_done_wait_sum_seconds * 1000,
                stats.read_done_wait_max_seconds * 1000,
                stats.barrier_seconds * 1000,
                stats.ready_sent,
                stats.ready_received,
                stats.read_done_attached,
                stats.read_done_received,
                stats.duplicate_notifications,
                stats.stale_notifications,
                stats.notification_poll_calls,
                stats.abort_sent,
                stats.abort_received,
                stats.abort_send_failures,
            )

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            for local_h, remote_h, xfer_h in self._xfer_entries:
                with contextlib.suppress(Exception):
                    self._nixl_wrapper.release_xfer_handle(xfer_h)
                with contextlib.suppress(Exception):
                    self._nixl_wrapper.release_dlist_handle(local_h)
                with contextlib.suppress(Exception):
                    self._nixl_wrapper.release_dlist_handle(remote_h)
            for entry in self._active_reads.values():
                self._release_active_read(entry)
        with contextlib.suppress(Exception):
            for descs in self._registered_descs:
                with contextlib.suppress(Exception):
                    self._nixl_wrapper.deregister_memory(descs)
            self._registered_descs.clear()
        with contextlib.suppress(Exception):
            for agent_name in self._remote_agents.values():
                with contextlib.suppress(Exception):
                    self._nixl_wrapper.remove_remote_agent(agent_name)
            self._remote_agents.clear()
            self._remote_agent_ranks.clear()


class PyNcclEplbCommunicator(EplbCommunicator):
    """EPLB communicator backed by PyNcclCommunicator using ncclSend/ncclRecv."""

    def __init__(
        self,
        pynccl_comm: PyNcclCommunicator,
        cuda_stream: torch.cuda.Stream | None = None,
    ) -> None:
        self._pynccl_comm = pynccl_comm
        self._cuda_stream = cuda_stream
        self._group_started = False
        self._log_initialized()

    def _ensure_group_started(self) -> None:
        if not self._group_started:
            self._pynccl_comm.group_start()
            self._group_started = True

    @property
    def perf_label(self) -> str:
        return "pynccl_reference"

    def add_send(
        self,
        tensors: list[torch.Tensor],
        dst_rank: int,
        expert_id: int,  # unused by this backend
    ) -> None:
        self._ensure_group_started()
        for tensor in tensors:
            self._pynccl_comm.send(tensor, dst_rank, stream=self._cuda_stream)

    def add_recv(
        self,
        tensors: list[torch.Tensor],
        src_rank: int,
        expert_id: int,  # unused by this backend
    ) -> None:
        self._ensure_group_started()
        for tensor in tensors:
            self._pynccl_comm.recv(tensor, src_rank, stream=self._cuda_stream)

    def execute(self) -> None:
        if self._group_started:
            stream = self._cuda_stream or torch.cuda.current_stream()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(stream)
            submit_started = time.perf_counter()
            try:
                self._pynccl_comm.group_end()
            finally:
                submit_ms = (time.perf_counter() - submit_started) * 1000
                end_event.record(stream)
                self._group_started = False
                self._set_last_backend_perf(
                    EplbBackendPerf(
                        fields={"nccl_group_submit_ms": submit_ms},
                        gpu_events=(start_event, end_event),
                    )
                )


def create_eplb_communicator(
    group_coordinator: GroupCoordinator,
    backend: str,
    expert_weights: Sequence[Sequence[torch.Tensor]],
    expert_buffer: Sequence[torch.Tensor],
    enable_nixl_sync_protocol: bool = False,
) -> EplbCommunicator:
    """Create an EPLB communicator for the given backend.

    Args:
        group_coordinator: Process-group coordinator that provides the
            device and CPU communication groups.
        backend: Communicator backend name (``"torch_nccl"``,
            ``"torch_gloo"``, ``"pynccl"``, or ``"nixl"``).
            Falls back to ``"torch_nccl"`` when *None*.
            Stateless (elastic EP) groups support ``"torch_nccl"``,
            ``"pynccl"``, and ``"nixl"``; ``"torch_nccl"`` is silently
            promoted to ``"pynccl"``.  ``"nixl"`` uses deferred remote
            agent setup to avoid collective deadlocks during elastic
            scaling.  When tensors reside on CPU, ``"torch_gloo"`` or
            ``"torch_nccl"`` are used via the CPU process group.
        expert_weights: Expert weight tensors for *all* MoE layers.
            Shape ``(num_layers)(num_tensors_per_layer)``.
            NixlEplbCommunicator registers all layers with NIXL for
            zero-copy RDMA reads.
        expert_buffer: Pre-allocated receive buffer tensors (one per
            weight tensor in a single layer).
        enable_nixl_sync_protocol: Enable the NIXL notification protocol for
            synchronous EPLB. Stateless elastic groups always use the legacy path.
    """
    first_layer = expert_weights[0] if expert_weights else []
    tensor_device_type = first_layer[0].device.type if first_layer else "cpu"
    torch_group = (
        group_coordinator.cpu_group
        if tensor_device_type == "cpu"
        else group_coordinator.device_group
    )

    def _create_pynccl() -> EplbCommunicator:
        if tensor_device_type == "cpu":
            raise RuntimeError(
                "EPLB communicator 'pynccl' supports only cuda-like devices "
                f"(got {tensor_device_type})."
            )
        unsupported_dtypes = sorted(
            {
                tensor.dtype
                for tensor in first_layer
                if not ncclDataTypeEnum.supports_torch_dtype(tensor.dtype)
            },
            key=str,
        )
        if unsupported_dtypes:
            raise RuntimeError(
                "EPLB communicator 'pynccl' requested but expert weights contain "
                "unsupported dtypes: "
                f"({', '.join(str(dtype) for dtype in unsupported_dtypes)})."
            )

        device_comm = group_coordinator.device_communicator
        pynccl_comm = (
            getattr(device_comm, "pynccl_comm", None)
            if device_comm is not None
            else None
        )
        if pynccl_comm is None or pynccl_comm.disabled or not pynccl_comm.available:
            raise RuntimeError("EPLB communicator 'pynccl' requested but unavailable.")
        try:
            return PyNcclEplbCommunicator(pynccl_comm=pynccl_comm)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize PyNcclEplbCommunicator ({exc})."
            ) from exc

    is_stateless = isinstance(group_coordinator, StatelessGroupCoordinator)
    if is_stateless:
        if backend == "nixl":
            pass  # handled below with defer_remote_setup=True
        elif backend not in ("torch_nccl", "pynccl"):
            raise ValueError(
                f"Elastic EP requires 'torch_nccl', 'pynccl', or 'nixl' "
                f"EPLB communicator (got '{backend}')."
            )
        else:
            if backend == "torch_nccl":
                logger.warning(
                    "Stateless elastic EP requires PyNCCL backend. "
                    "Forcing EPLB communicator to 'pynccl'."
                )
                backend = "pynccl"
            return _create_pynccl()

    if backend == "nixl":
        if not has_nixl():
            raise RuntimeError(
                "EPLB communicator 'nixl' requested but NIXL is unavailable."
            )
        if not (current_platform.is_cuda_alike() and tensor_device_type != "cpu"):
            raise RuntimeError(
                "EPLB communicator 'nixl' supports only cuda-like devices "
                f"(got {tensor_device_type})."
            )
        try:
            return NixlEplbCommunicator(
                cpu_group=group_coordinator.cpu_group,
                all_expert_weights=expert_weights,
                expert_buffer=expert_buffer,
                defer_remote_setup=is_stateless,
                enable_sync_protocol=(enable_nixl_sync_protocol and not is_stateless),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize NixlEplbCommunicator ({exc})."
            ) from exc
    elif backend == "torch_gloo":
        return TorchDistGlooStagedEplbCommunicator(
            cpu_group=group_coordinator.cpu_group,
        )
    elif backend == "torch_nccl":
        return TorchDistNcclEplbCommunicator(ep_group=torch_group)
    elif backend == "pynccl":
        return _create_pynccl()
    raise ValueError(f"Unknown EPLB communicator backend: {backend}")
