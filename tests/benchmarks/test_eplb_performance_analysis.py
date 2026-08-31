# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import csv
import json
from pathlib import Path

from benchmarks.eplb.analyze_performance import (
    AnalysisConfig,
    analyze,
    parse_client_log,
    parse_log,
    summarize_generations,
    validate_client_consistency,
    validate_log,
)


def _perf_line(
    communicator: str,
    rank: int,
    rearrangement_id: int = 0,
) -> str:
    common = (
        f"communicator={communicator} rearrangement_id={rearrangement_id} "
        f"generation_id={rearrangement_id} "
        f"layer_id=0 rank={rank} "
    )
    if rank == 0:
        common += (
            "generation_wall_ms=10 staging_ms=1 tx_bytes=100 rx_bytes=0 "
            "io_bytes=100 send_transfers=1 recv_transfers=0 "
        )
    else:
        common += (
            "generation_wall_ms=12 staging_ms=1.5 tx_bytes=0 rx_bytes=100 "
            "io_bytes=100 send_transfers=0 recv_transfers=1 "
        )
    if communicator == "baseline_nixl":
        return common + (
            f"backend_wall_ms={8 + rank} transfer_wait_ms={7 + rank} "
            f"barrier_ms={1 + rank}"
        )
    if communicator == "candidate_nixl":
        return common + (
            f"backend_wall_ms={8 + rank} ready_wait_ms=1 "
            f"read_execution_ms={4 + rank} read_done_wait_ms=1 "
            f"protocol_residual_ms={2 + rank}"
        )
    return common + (
        f"nccl_group_submit_ms={0.2 + rank / 10} nccl_group_gpu_ms={5 + rank}"
    )


def _read_line(communicator: str, rearrangement_id: int = 0) -> str:
    return (
        f"communicator={communicator} rearrangement_id={rearrangement_id} "
        f"generation_id={rearrangement_id} "
        "layer_id=0 rank=1 source_rank=0 expert_id=7 payload_bytes=100"
    )


def _write_log(
    path: Path,
    communicator: str,
    rearrangement_ids: tuple[int, ...] = (0,),
) -> None:
    lines = []
    for rearrangement_id in rearrangement_ids:
        lines.extend(
            [
                "INFO Rearranged experts in 0.123 s.",
                f"INFO EPLB_PERF {_perf_line(communicator, 0, rearrangement_id)}",
                f"INFO EPLB_PERF {_perf_line(communicator, 1, rearrangement_id)}",
            ]
        )
        if communicator != "pynccl_reference":
            lines.append(f"INFO EPLB_READ {_read_line(communicator, rearrangement_id)}")
        if communicator == "candidate_nixl":
            for rank in range(2):
                lines.append(
                    "DEBUG NIXL EPLB execute stats: "
                    f"rank={rank} generation={rearrangement_id} "
                    "sync_protocol=True ready_sent=1 ready_received=1 "
                    "read_done_attached=1 read_done_received=1 "
                    "duplicate_notifications=0 stale_notifications=0 "
                    "abort_sent=0 abort_received=0 abort_send_failures=0"
                )
    path.write_text("\n".join(lines) + "\n")


def _write_client_log(path: Path, prompt_hash: str = "abc123") -> None:
    record = {
        "phase": 0,
        "seed": 42,
        "prompt_len": 1024,
        "prompt_sha256": prompt_hash,
        "requests": 32,
        "successful_requests": 32,
        "start_rearrangements": 1,
        "end_rearrangements": 2,
        "new_rearrangements": 1,
        "phase_elapsed_s": 2.5,
        "request_latency_mean_ms": 100,
        "request_latency_median_ms": 95,
        "request_latency_p95_ms": 120,
        "request_latency_max_ms": 130,
    }
    path.write_text(json.dumps(record) + "\n")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def test_analyze_exports_scopes_and_flags_only_unaccounted_residual(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline.log"
    candidate = tmp_path / "candidate.log"
    pynccl = tmp_path / "pynccl.log"
    _write_log(baseline, "baseline_nixl")
    _write_log(candidate, "candidate_nixl")
    _write_log(pynccl, "pynccl_reference")
    baseline_client = tmp_path / "baseline-client.log"
    candidate_client = tmp_path / "candidate-client.log"
    pynccl_client = tmp_path / "pynccl-client.log"
    _write_client_log(baseline_client)
    _write_client_log(candidate_client)
    _write_client_log(pynccl_client)
    output_dir = tmp_path / "analysis"

    report, outputs = analyze(
        AnalysisConfig(
            baseline_log=baseline,
            candidate_log=candidate,
            pynccl_log=pynccl,
            baseline_client_log=baseline_client,
            candidate_client_log=candidate_client,
            pynccl_client_log=pynccl_client,
            output_dir=output_dir,
            expected_ranks=2,
            min_rearrangements=1,
            expected_phases=1,
            steady_state_start_rearrangement=0,
        )
    )

    assert report.status == "PASS"
    assert len(outputs) == 9
    validation = json.loads((output_dir / "validation_report.json").read_text())
    assert validation["residual_flag_count"] == 0
    assert validation["residual_flag_counts"] == {
        "all": 0,
        "initialization": 0,
        "steady_state": 0,
    }

    generations = _read_csv(output_dir / "generation_summary.csv")
    candidate_generation = next(
        row for row in generations if row["communicator"] == "candidate_nixl"
    )
    assert candidate_generation["critical_rank"] == "1"
    assert float(candidate_generation["decomposition_residual_ms"]) == 3.5
    assert float(candidate_generation["unaccounted_residual_ms"]) == 0.5
    assert candidate_generation["residual_flag"] == ""

    comparison = _read_csv(output_dir / "communicator_comparison.csv")
    inner_sources = {
        row["communicator"]: row["inner_time_source"]
        for row in comparison
        if row["scope"] == "steady_state"
    }
    assert inner_sources == {
        "baseline_nixl": "transfer_wait_ms",
        "candidate_nixl": "read_execution_ms",
        "pynccl_reference": "nccl_group_gpu_ms",
    }
    assert {row["scope"] for row in comparison} == {"all", "steady_state"}
    assert all(float(row["total_io_mean_bytes"]) == 200 for row in comparison)
    deltas = _read_csv(output_dir / "comparison_deltas.csv")
    candidate_outer = next(
        row
        for row in deltas
        if row["scope"] == "steady_state"
        and row["comparison"] == "candidate_vs_baseline"
        and row["metric"] == "rearrangement_wall_mean_ms"
    )
    assert float(candidate_outer["percent_change"]) == 0
    assert (output_dir / "client_phase_records.csv").is_file()
    phase_rows = _read_csv(output_dir / "candidate_phase_summary.csv")
    assert {row["phase"] for row in phase_rows} == {
        "ready_wait_ms",
        "read_execution_ms",
        "read_done_wait_ms",
        "protocol_residual_ms",
        "staging_ms",
        "decomposition_residual_ms",
        "decomposition_residual_pct",
        "unaccounted_residual_ms",
        "unaccounted_residual_pct",
    }
    assert {row["scope"] for row in phase_rows} == {"all", "steady_state"}


def test_analyze_separates_initialization_and_steady_state(tmp_path: Path) -> None:
    logs = {}
    for label, communicator in (
        ("baseline", "baseline_nixl"),
        ("candidate", "candidate_nixl"),
        ("pynccl", "pynccl_reference"),
    ):
        logs[label] = tmp_path / f"{label}.log"
        _write_log(logs[label], communicator, rearrangement_ids=(0, 1))

    output_dir = tmp_path / "analysis"
    report, _ = analyze(
        AnalysisConfig(
            baseline_log=logs["baseline"],
            candidate_log=logs["candidate"],
            pynccl_log=logs["pynccl"],
            output_dir=output_dir,
            expected_ranks=2,
            min_rearrangements=2,
            steady_state_start_rearrangement=1,
        )
    )

    assert report.status == "PASS_WITH_WARNINGS"
    comparison = _read_csv(output_dir / "communicator_comparison.csv")
    assert {row["scope"] for row in comparison} == {
        "all",
        "initialization",
        "steady_state",
    }
    counts = {
        (row["scope"], row["communicator"]): int(row["generation_count"])
        for row in comparison
    }
    assert counts[("initialization", "candidate_nixl")] == 1
    assert counts[("steady_state", "candidate_nixl")] == 1
    assert counts[("all", "candidate_nixl")] == 2


def test_validation_rejects_missing_rank_and_unbalanced_bytes(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "baseline.log"
    log_path.write_text(
        "INFO Rearranged experts in 0.1 s.\n"
        f"INFO EPLB_PERF {_perf_line('baseline_nixl', 0)}\n"
        f"INFO EPLB_READ {_read_line('baseline_nixl')}\n"
    )

    report = validate_log(
        parse_log("baseline", log_path),
        expected_ranks=2,
        min_rearrangements=1,
    )

    assert report.status == "FAIL"
    assert any("expected [0, 1]" in error for error in report.errors)
    assert any("unbalanced bytes" in error for error in report.errors)
    assert any("no matching performance record" in error for error in report.errors)


def test_candidate_summary_uses_critical_rank_phases(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.log"
    _write_log(candidate, "candidate_nixl")
    parsed = parse_log("candidate", candidate)

    summaries = summarize_generations(
        parsed,
        residual_threshold_pct=10,
        negative_residual_tolerance_ms=0.1,
    )

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.critical_rank == 1
    assert summary.generation_wall_ms == 12
    assert summary.read_execution_ms == 5
    assert summary.staging_ms == 1.5
    assert summary.decomposition_residual_ms == 3.5
    assert summary.unaccounted_residual_ms == 0.5
    assert summary.residual_flag == ""
    assert summary.rank_cov == 1 / 11


def test_validation_reports_malformed_numeric_field(tmp_path: Path) -> None:
    log_path = tmp_path / "baseline.log"
    malformed = _perf_line("baseline_nixl", 0).replace(
        "tx_bytes=100",
        "tx_bytes=invalid",
    )
    log_path.write_text(
        "INFO Rearranged experts in 0.1 s.\n"
        f"INFO EPLB_PERF {malformed}\n"
        f"INFO EPLB_PERF {_perf_line('baseline_nixl', 1)}\n"
        f"INFO EPLB_READ {_read_line('baseline_nixl')}\n"
    )

    report = validate_log(
        parse_log("baseline", log_path),
        expected_ranks=2,
        min_rearrangements=1,
    )

    assert report.status == "FAIL"
    assert any("tx_bytes must be an integer" in error for error in report.errors)


def test_client_validation_rejects_prompt_mismatch(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline-client.log"
    candidate = tmp_path / "candidate-client.log"
    pynccl = tmp_path / "pynccl-client.log"
    _write_client_log(baseline)
    _write_client_log(candidate, prompt_hash="different")
    _write_client_log(pynccl)

    report = validate_client_consistency(
        [
            parse_client_log("baseline", baseline),
            parse_client_log("candidate", candidate),
            parse_client_log("pynccl", pynccl),
        ]
    )

    assert report.status == "FAIL"
    assert "prompt identity differs" in report.errors[0]
