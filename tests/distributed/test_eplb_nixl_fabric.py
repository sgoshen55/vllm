# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-rank CPU simulation of the NIXL EPLB sync protocol.

N ranks run the real rearrangement plan in threads over a fabric that moves
real bytes between registered tensors. No GPU or NIXL needed.
"""

import random
import threading
import time
from collections.abc import Callable
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm.distributed.eplb.eplb_communicator import (
    NixlEplbCommunicator,
    NixlEplbNotification,
    NixlEplbTransferTracker,
)
from vllm.distributed.eplb.rebalance_execute import rearrange_expert_weights_inplace

from .test_eplb_execute import (
    create_expert_indices_with_redundancy,
    create_expert_weights,
    create_redundancy_config,
    verify_expert_weights_after_shuffle,
)
from .test_eplb_nixl_protocol import NO_OP_STREAM, agent_name, make_communicator

SEEDS = [0, 1, 2]
HIDDEN_SIZES = [16, 32]
JOIN_TIMEOUT_SECONDS = 30.0
MISMATCH_MESSAGES = ("undeclared reader", "without matching add_recv", "timed out")


class Fabric:
    """Shared transport for the FabricAgents of one simulated cluster.

    READs copy real bytes between registered tensors. Notifications land in
    per-agent inboxes grouped by sender name.
    """

    def __init__(self, seed: int) -> None:
        self.lock = threading.Lock()
        self.rng = random.Random(seed)
        self.registry: list[tuple[int, int, torch.Tensor]] = []
        self.inboxes: dict[str, dict[str, list[bytes]]] = {}
        self.transfers: list[tuple[str, str, bytes]] = []
        self.notifications: list[tuple[str, str, bytes]] = []

    def register(self, tensor: torch.Tensor) -> None:
        start = tensor.data_ptr()
        flat = tensor.view(torch.uint8).reshape(-1)
        self.registry.append((start, start + tensor.nbytes, flat))

    def view(self, address: int, nbytes: int) -> torch.Tensor:
        for start, end, flat in self.registry:
            if start <= address and address + nbytes <= end:
                return flat[address - start : address - start + nbytes]
        raise KeyError(f"unregistered range {address:#x}+{nbytes}")

    def draw_completion_delay(self) -> int:
        with self.lock:
            return self.rng.randint(0, 3)

    def deliver(self, sender: str, receiver: str, payload: bytes) -> None:
        with self.lock:
            self.inboxes[receiver].setdefault(sender, []).append(payload)
            self.notifications.append((sender, receiver, payload))

    def drain(self, receiver: str) -> dict[str, list[bytes]]:
        with self.lock:
            inbox = self.inboxes[receiver]
            self.inboxes[receiver] = {}
            return inbox


class FabricAgent:
    def __init__(self, fabric: Fabric, name: str) -> None:
        self.fabric = fabric
        self.name = name
        self._dlists: dict[int, tuple[str, list[tuple[int, int, int]]]] = {}
        self._xfers: dict[int, SimpleNamespace] = {}
        self._next_handle = 1
        fabric.inboxes[name] = {}

    def _new_handle(self) -> int:
        handle = self._next_handle
        self._next_handle += 1
        return handle

    def get_xfer_descs(self, descs, memory_type):
        return descs

    def prep_xfer_dlist(self, agent, descs) -> int:
        handle = self._new_handle()
        self._dlists[handle] = (agent, list(descs))
        return handle

    def make_prepped_xfer(
        self,
        operation,
        local_handle,
        local_indices,
        remote_handle,
        remote_indices,
        notif_msg=b"",
    ) -> int:
        assert operation == "READ"
        _, local_descs = self._dlists[local_handle]
        remote_agent, remote_descs = self._dlists[remote_handle]
        handle = self._new_handle()
        self._xfers[handle] = SimpleNamespace(
            pairs=[
                (local_descs[int(i)], remote_descs[int(j)])
                for i, j in zip(local_indices, remote_indices)
            ],
            remote_agent=remote_agent,
            notif_msg=notif_msg,
            checks_left=self.fabric.draw_completion_delay(),
            done=False,
        )
        return handle

    def transfer(self, handle: int) -> str:
        xfer = self._xfers[handle]
        if xfer.checks_left == 0:
            self._complete(xfer)
            return "DONE"
        return "PROC"

    def check_xfer_state(self, handle: int) -> str:
        xfer = self._xfers[handle]
        if xfer.done:
            return "DONE"
        xfer.checks_left -= 1
        if xfer.checks_left == 0:
            self._complete(xfer)
            return "DONE"
        return "PROC"

    def _complete(self, xfer: SimpleNamespace) -> None:
        # Copy first, then notify: NIXL emits the attached notification only
        # after the READ has completed.
        for (local_addr, nbytes, _), (remote_addr, remote_nbytes, _) in xfer.pairs:
            assert nbytes == remote_nbytes
            self.fabric.view(local_addr, nbytes).copy_(
                self.fabric.view(remote_addr, nbytes)
            )
        xfer.done = True
        with self.fabric.lock:
            self.fabric.transfers.append((self.name, xfer.remote_agent, xfer.notif_msg))
        if xfer.notif_msg:
            self.fabric.deliver(self.name, xfer.remote_agent, xfer.notif_msg)

    def send_notif(self, agent: str, notif_msg: bytes) -> None:
        self.fabric.deliver(self.name, agent, notif_msg)

    def get_new_notifs(self) -> dict[str, list[bytes]]:
        return self.fabric.drain(self.name)

    def release_xfer_handle(self, handle: int) -> None:
        self._xfers.pop(handle, None)

    def release_dlist_handle(self, handle: int) -> None:
        self._dlists.pop(handle, None)


def new_placement(cluster: SimpleNamespace) -> torch.Tensor:
    total = cluster.world_size * cluster.num_local_experts
    return create_expert_indices_with_redundancy(
        cluster.num_layers,
        cluster.num_logical_experts,
        total,
        create_redundancy_config(cluster.num_logical_experts, total),
    )


def make_cluster(
    seed: int,
    *,
    world_size: int = 4,
    num_local_experts: int = 2,
    num_layers: int = 3,
    num_logical_experts: int = 5,
    protocol: bool = True,
    clock: Callable[[], float] | None = None,
    timeout_seconds: float = 300.0,
) -> SimpleNamespace:
    random.seed(seed)
    torch.manual_seed(seed)
    cluster = SimpleNamespace(
        world_size=world_size,
        num_local_experts=num_local_experts,
        num_layers=num_layers,
        num_logical_experts=num_logical_experts,
        fabric=Fabric(seed),
    )
    cluster.indices = new_placement(cluster)
    cluster.weights = [
        create_expert_weights(
            num_layers,
            num_local_experts,
            HIDDEN_SIZES,
            rank,
            torch.device("cpu"),
            cluster.indices,
        )
        for rank in range(world_size)
    ]
    cluster.buffers = [
        [torch.empty_like(w) for w in cluster.weights[rank][0]]
        for rank in range(world_size)
    ]
    for rank in range(world_size):
        for layer_tensors in cluster.weights[rank]:
            for tensor in layer_tensors:
                cluster.fabric.register(tensor)
        for tensor in cluster.buffers[rank]:
            cluster.fabric.register(tensor)
    send_meta = {
        rank: {
            (layer, t_idx): (tensor.data_ptr(), tensor.nbytes // num_local_experts, 0)
            for layer, layer_tensors in enumerate(cluster.weights[rank])
            for t_idx, tensor in enumerate(layer_tensors)
        }
        for rank in range(world_size)
    }
    cluster.communicators = [
        make_communicator(
            rank,
            FabricAgent(cluster.fabric, agent_name(rank)),
            protocol=protocol,
            clock=clock,
            timeout_seconds=timeout_seconds,
            world_size=world_size,
            num_local_experts=num_local_experts,
            remote_send_meta={
                peer: send_meta[peer] for peer in range(world_size) if peer != rank
            },
        )
        for rank in range(world_size)
    ]
    return cluster


def run_rearrangement(
    cluster: SimpleNamespace,
    new_indices: torch.Tensor,
    *,
    per_rank: dict[int, torch.Tensor] | None = None,
    raise_errors: bool = True,
) -> list[BaseException | None]:
    errors: list[BaseException | None] = [None] * cluster.world_size

    def worker(rank: int) -> None:
        try:
            rearrange_expert_weights_inplace(
                cluster.indices,
                (per_rank or {}).get(rank, new_indices),
                cluster.weights[rank],
                cluster.buffers[rank],
                SimpleNamespace(size=lambda: cluster.world_size, rank=lambda: rank),
                cluster.communicators[rank],
            )
        except BaseException as exc:
            errors[rank] = exc

    threads = [
        threading.Thread(target=worker, args=(rank,), daemon=True)
        for rank in range(cluster.world_size)
    ]
    # Patch the stream once around the whole run; patching per thread races.
    with mock.patch.object(
        torch.accelerator, "current_stream", return_value=NO_OP_STREAM
    ):
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + JOIN_TIMEOUT_SECONDS
        for thread in threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
    if any(thread.is_alive() for thread in threads):
        pytest.fail(f"rank threads still alive after {JOIN_TIMEOUT_SECONDS}s")
    if raise_errors:
        for error in errors:
            if error is not None:
                raise error
        cluster.indices = new_indices
    return errors


def verify_cluster(cluster: SimpleNamespace, new_indices: torch.Tensor) -> None:
    for rank in range(cluster.world_size):
        assert verify_expert_weights_after_shuffle(
            cluster.weights[rank],
            new_indices,
            HIDDEN_SIZES,
            rank,
            cluster.num_local_experts,
        ), f"rank {rank} holds wrong weights"


def assert_quiescent(cluster: SimpleNamespace, layers_run: int) -> None:
    for communicator in cluster.communicators:
        assert communicator._tracker.active_generation is None
        assert communicator._tracker.completed_generation == layers_run - 1
    assert all(not inbox for inbox in cluster.fabric.inboxes.values())


def run_rounds(cluster: SimpleNamespace, rounds: int) -> None:
    for round_idx in range(rounds):
        new_indices = new_placement(cluster)
        run_rearrangement(cluster, new_indices)
        verify_cluster(cluster, new_indices)
        assert_quiescent(cluster, layers_run=(round_idx + 1) * cluster.num_layers)


def delay_layer_start(communicator: NixlEplbCommunicator, rng: random.Random):
    original = communicator.set_transfer_context

    def delayed(old_indices, layer_idx: int) -> None:
        time.sleep(rng.uniform(0.0, 0.020))
        original(old_indices, layer_idx)

    communicator.set_transfer_context = delayed


def count_early_ready(tracker: NixlEplbTransferTracker, counts: list[int]) -> None:
    original = tracker.record_ready

    def counting(key, sender: int) -> None:
        if key.generation != tracker.active_generation:
            counts[0] += 1
        original(key, sender)

    tracker.record_ready = counting


@pytest.mark.parametrize("seed", SEEDS)
def test_fabric_matches_legacy_barrier_path(seed: int) -> None:
    cluster = make_cluster(seed, protocol=False)
    barrier = threading.Barrier(cluster.world_size)
    for communicator in cluster.communicators:
        communicator._post_read_barrier = barrier.wait
    new_indices = new_placement(cluster)
    run_rearrangement(cluster, new_indices)
    verify_cluster(cluster, new_indices)


@pytest.mark.parametrize("seed", SEEDS)
def test_protocol_rearranges_correctly(seed: int) -> None:
    run_rounds(make_cluster(seed), rounds=3)


def test_protocol_under_skew() -> None:
    early_ready = [0]
    for seed in SEEDS:
        cluster = make_cluster(seed)
        for rank, communicator in enumerate(cluster.communicators):
            delay_layer_start(communicator, random.Random(seed * 100 + rank))
            count_early_ready(communicator._tracker, early_ready)
        run_rounds(cluster, rounds=3)
    assert early_ready[0] > 0, "no READY arrived before its generation opened"


@pytest.mark.parametrize("seed", SEEDS)
def test_protocol_with_no_transfers_in_a_layer(seed: int) -> None:
    cluster = make_cluster(seed)
    new_indices = new_placement(cluster)
    new_indices[1] = cluster.indices[1]
    run_rearrangement(cluster, new_indices)
    verify_cluster(cluster, new_indices)
    assert_quiescent(cluster, layers_run=cluster.num_layers)
    generations = {
        NixlEplbNotification.decode(payload).key.generation
        for _, _, payload in cluster.fabric.transfers + cluster.fabric.notifications
    }
    assert generations and 1 not in generations


@pytest.mark.parametrize("seed", SEEDS)
def test_plan_mismatch_is_detected(seed: int) -> None:
    cluster = make_cluster(
        seed, num_layers=1, clock=time.monotonic, timeout_seconds=2.0
    )
    new_indices = new_placement(cluster)
    rank0_rows = slice(0, cluster.num_local_experts)
    excluded = set(cluster.indices[0, rank0_rows].tolist()) | set(
        new_indices[0, rank0_rows].tolist()
    )
    corrupted = new_indices.clone()
    corrupted[0, 0] = next(
        expert
        for expert in range(cluster.num_logical_experts)
        if expert not in excluded
    )
    errors = run_rearrangement(
        cluster, new_indices, per_rank={0: corrupted}, raise_errors=False
    )
    messages = [str(error) for error in errors if isinstance(error, RuntimeError)]
    assert any(
        any(fragment in message for fragment in MISMATCH_MESSAGES)
        for message in messages
    ), errors
