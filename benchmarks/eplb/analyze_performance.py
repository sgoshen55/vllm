# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate and analyze EPLB performance logs from three communicators."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TypeAlias

Scalar: TypeAlias = bool | float | int | str
GenerationKey: TypeAlias = tuple[int, int, int]
RankKey: TypeAlias = tuple[int, int, int, int]

PERF_MARKER = "EPLB_PERF "
READ_MARKER = "EPLB_READ "
PROTOCOL_MARKER = "NIXL EPLB execute stats:"
REARRANGEMENT_PATTERN = re.compile(
    r"Rearranged experts\s+in\s+([0-9]+(?:\.[0-9]+)?)\s+s\."
)
FATAL_PATTERN = re.compile(
    r" ERROR |Traceback|RuntimeError|NIXL_ERR|NCCL WARN|"
    r"Worker proc failed|WorkerProc failed|Engine core initialization failed|"
    r"CUDA out of memory"
)

COMMON_PERF_FIELDS = {
    "communicator",
    "rearrangement_id",
    "generation_id",
    "layer_id",
    "rank",
    "generation_wall_ms",
    "staging_ms",
    "tx_bytes",
    "rx_bytes",
    "io_bytes",
    "send_transfers",
    "recv_transfers",
}
BACKEND_PERF_FIELDS = {
    "baseline_nixl": {"backend_wall_ms", "transfer_wait_ms", "barrier_ms"},
    "candidate_nixl": {
        "backend_wall_ms",
        "ready_wait_ms",
        "read_execution_ms",
        "read_done_wait_ms",
        "protocol_residual_ms",
    },
    "pynccl_reference": {"nccl_group_submit_ms", "nccl_group_gpu_ms"},
}
READ_FIELDS = {
    "communicator",
    "rearrangement_id",
    "generation_id",
    "layer_id",
    "rank",
    "source_rank",
    "expert_id",
    "payload_bytes",
}
PROTOCOL_FIELDS = {
    "rank",
    "generation",
    "sync_protocol",
    "ready_sent",
    "ready_received",
    "read_done_attached",
    "read_done_received",
    "duplicate_notifications",
    "stale_notifications",
    "abort_sent",
    "abort_received",
    "abort_send_failures",
}
CLIENT_PHASE_FIELDS = {
    "phase",
    "seed",
    "prompt_len",
    "prompt_sha256",
    "requests",
    "successful_requests",
    "start_rearrangements",
    "end_rearrangements",
    "new_rearrangements",
    "phase_elapsed_s",
    "request_latency_mean_ms",
    "request_latency_median_ms",
    "request_latency_p95_ms",
    "request_latency_max_ms",
}
EXPECTED_COMMUNICATORS = {
    "baseline": "baseline_nixl",
    "candidate": "candidate_nixl",
    "pynccl": "pynccl_reference",
}
INNER_TIME_FIELDS = {
    "baseline_nixl": "transfer_wait_ms",
    "candidate_nixl": "read_execution_ms",
    "pynccl_reference": "nccl_group_gpu_ms",
}


class LogAnalysisError(RuntimeError):
    """Raised when a log cannot be parsed or analyzed."""


@dataclass(frozen=True, slots=True)
class PerfRecord:
    values: dict[str, Scalar]
    line_number: int

    @property
    def communicator(self) -> str:
        return str(self.values["communicator"])

    @property
    def rank_key(self) -> RankKey:
        return (
            self.integer("rearrangement_id"),
            self.integer("generation_id"),
            self.integer("layer_id"),
            self.integer("rank"),
        )

    @property
    def generation_key(self) -> GenerationKey:
        return self.rank_key[:3]

    def integer(self, name: str) -> int:
        value = self.values[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise LogAnalysisError(
                f"line {self.line_number}: {name} must be an integer, got {value!r}"
            )
        return value

    def number(self, name: str) -> float:
        value = self.values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise LogAnalysisError(
                f"line {self.line_number}: {name} must be numeric, got {value!r}"
            )
        return float(value)


@dataclass(frozen=True, slots=True)
class ReadRecord:
    values: dict[str, Scalar]
    line_number: int

    @property
    def rank_key(self) -> RankKey:
        return (
            self.integer("rearrangement_id"),
            self.integer("generation_id"),
            self.integer("layer_id"),
            self.integer("rank"),
        )

    def integer(self, name: str) -> int:
        value = self.values[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise LogAnalysisError(
                f"line {self.line_number}: {name} must be an integer, got {value!r}"
            )
        return value


@dataclass(slots=True)
class ParsedLog:
    label: str
    path: Path
    perf_records: list[PerfRecord] = field(default_factory=list)
    read_records: list[ReadRecord] = field(default_factory=list)
    protocol_records: list[dict[str, Scalar]] = field(default_factory=list)
    rearrangement_ms: list[float] = field(default_factory=list)
    fatal_lines: list[tuple[int, str]] = field(default_factory=list)

    @property
    def expected_communicator(self) -> str:
        return EXPECTED_COMMUNICATORS[self.label]


@dataclass(frozen=True, slots=True)
class ClientPhaseRecord:
    values: dict[str, Scalar]
    line_number: int

    def integer(self, name: str) -> int:
        value = self.values[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise LogAnalysisError(
                f"line {self.line_number}: {name} must be an integer, got {value!r}"
            )
        return value

    def number(self, name: str) -> float:
        value = self.values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise LogAnalysisError(
                f"line {self.line_number}: {name} must be numeric, got {value!r}"
            )
        return float(value)


@dataclass(slots=True)
class ParsedClientLog:
    label: str
    path: Path
    phases: list[ClientPhaseRecord] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class GenerationSummary:
    communicator: str
    rearrangement_id: int
    generation_id: int
    layer_id: int
    rank_count: int
    critical_rank: int
    generation_wall_ms: float
    staging_ms: float
    inner_time_source: str
    inner_time_ms: float
    tx_bytes_total: int
    rx_bytes_total: int
    unique_payload_bytes: int
    rank_cov: float
    bytes_balanced: bool
    ready_wait_ms: float | None = None
    read_execution_ms: float | None = None
    read_done_wait_ms: float | None = None
    protocol_residual_ms: float | None = None
    residual_ms: float | None = None
    residual_pct: float | None = None
    residual_flag: str = ""


@dataclass(slots=True)
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.errors:
            return "FAIL"
        if self.warnings:
            return "PASS_WITH_WARNINGS"
        return "PASS"


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    baseline_log: Path
    candidate_log: Path
    pynccl_log: Path
    output_dir: Path
    baseline_client_log: Path | None = None
    candidate_client_log: Path | None = None
    pynccl_client_log: Path | None = None
    expected_ranks: int = 8
    min_rearrangements: int = 64
    expected_phases: int = 64
    residual_threshold_pct: float = 10.0
    negative_residual_tolerance_ms: float = 0.1


def _coerce_value(value: str) -> Scalar:
    if value == "True":
        return True
    if value == "False":
        return False
    if re.fullmatch(r"-?[0-9]+", value):
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def _parse_fields(payload: str, path: Path, line_number: int) -> dict[str, Scalar]:
    fields: dict[str, Scalar] = {}
    for token in payload.split():
        if "=" not in token:
            continue
        name, value = token.split("=", 1)
        if not name or not value:
            raise LogAnalysisError(
                f"{path}:{line_number}: malformed key-value token {token!r}"
            )
        if name in fields:
            raise LogAnalysisError(f"{path}:{line_number}: duplicate field {name!r}")
        fields[name] = _coerce_value(value.rstrip(","))
    return fields


def parse_log(label: str, path: Path) -> ParsedLog:
    """Parse timer, READ, protocol, and rearrangement records from one log."""
    if label not in EXPECTED_COMMUNICATORS:
        raise ValueError(f"unknown log label: {label}")
    if not path.is_file():
        raise LogAnalysisError(f"log does not exist: {path}")

    parsed = ParsedLog(label=label, path=path)
    with path.open(errors="replace") as source:
        for line_number, line in enumerate(source, start=1):
            if PERF_MARKER in line:
                payload = line.split(PERF_MARKER, 1)[1]
                parsed.perf_records.append(
                    PerfRecord(_parse_fields(payload, path, line_number), line_number)
                )
            if READ_MARKER in line:
                payload = line.split(READ_MARKER, 1)[1]
                parsed.read_records.append(
                    ReadRecord(_parse_fields(payload, path, line_number), line_number)
                )
            if PROTOCOL_MARKER in line:
                payload = line.split(PROTOCOL_MARKER, 1)[1]
                parsed.protocol_records.append(
                    _parse_fields(payload, path, line_number)
                )
            match = REARRANGEMENT_PATTERN.search(line)
            if match and "profile" not in line.lower():
                parsed.rearrangement_ms.append(float(match.group(1)) * 1000)
            if FATAL_PATTERN.search(line):
                parsed.fatal_lines.append((line_number, line.strip()))
    return parsed


def parse_client_log(label: str, path: Path) -> ParsedClientLog:
    """Parse JSON phase summaries emitted by the skew-routing client."""
    if label not in EXPECTED_COMMUNICATORS:
        raise ValueError(f"unknown client log label: {label}")
    if not path.is_file():
        raise LogAnalysisError(f"client log does not exist: {path}")

    parsed = ParsedClientLog(label=label, path=path)
    with path.open(errors="replace") as source:
        for line_number, line in enumerate(source, start=1):
            stripped = line.strip()
            if not stripped.startswith("{"):
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(value, dict) or "phase" not in value:
                continue
            fields: dict[str, Scalar] = {}
            for name, field_value in value.items():
                if not isinstance(field_value, (bool, float, int, str)):
                    raise LogAnalysisError(
                        f"{path}:{line_number}: unsupported client field "
                        f"{name}={field_value!r}"
                    )
                fields[str(name)] = field_value
            parsed.phases.append(ClientPhaseRecord(fields, line_number))
    return parsed


def _missing_fields(values: dict[str, Scalar], required: set[str]) -> set[str]:
    return required - values.keys()


def validate_log(
    parsed: ParsedLog,
    expected_ranks: int,
    min_rearrangements: int,
) -> ValidationReport:
    """Validate one parsed communicator log."""
    report = ValidationReport()
    communicator = parsed.expected_communicator
    required_perf = COMMON_PERF_FIELDS | BACKEND_PERF_FIELDS[communicator]

    if not parsed.perf_records:
        report.errors.append(f"{parsed.label}: no {PERF_MARKER.strip()} records")
        return report
    if len(parsed.rearrangement_ms) < min_rearrangements:
        report.errors.append(
            f"{parsed.label}: found {len(parsed.rearrangement_ms)} rearrangements; "
            f"expected at least {min_rearrangements}"
        )
    for line_number, line in parsed.fatal_lines:
        report.errors.append(f"{parsed.label}:{line_number}: fatal pattern: {line}")

    records_by_generation: dict[GenerationKey, list[PerfRecord]] = defaultdict(list)
    rank_keys: set[RankKey] = set()
    for record in parsed.perf_records:
        missing = _missing_fields(record.values, required_perf)
        if missing:
            report.errors.append(
                f"{parsed.label}:{record.line_number}: missing fields {sorted(missing)}"
            )
            continue
        if record.communicator != communicator:
            report.errors.append(
                f"{parsed.label}:{record.line_number}: communicator is "
                f"{record.communicator!r}, expected {communicator!r}"
            )
        try:
            rank_key = record.rank_key
            for name in ("tx_bytes", "rx_bytes", "io_bytes"):
                record.integer(name)
            for name in ("send_transfers", "recv_transfers"):
                record.integer(name)
            for name in sorted(
                field for field in required_perf if field.endswith("_ms")
            ):
                if record.number(name) < 0:
                    report.errors.append(
                        f"{parsed.label}:{record.line_number}: {name} is negative"
                    )
            for name in (
                "tx_bytes",
                "rx_bytes",
                "io_bytes",
                "send_transfers",
                "recv_transfers",
            ):
                if record.integer(name) < 0:
                    report.errors.append(
                        f"{parsed.label}:{record.line_number}: {name} is negative"
                    )
            if not math.isclose(
                record.number("io_bytes"),
                record.number("tx_bytes") + record.number("rx_bytes"),
            ):
                report.errors.append(
                    f"{parsed.label}:{record.line_number}: io_bytes does not equal "
                    "tx_bytes + rx_bytes"
                )
        except (KeyError, LogAnalysisError) as exc:
            report.errors.append(f"{parsed.label}: {exc}")
            continue
        if rank_key in rank_keys:
            report.errors.append(
                f"{parsed.label}:{record.line_number}: duplicate rank record {rank_key}"
            )
            continue
        rank_keys.add(rank_key)
        records_by_generation[record.generation_key].append(record)

    expected_rank_set = set(range(expected_ranks))
    for key, records in sorted(records_by_generation.items()):
        ranks = {record.integer("rank") for record in records}
        if ranks != expected_rank_set:
            report.errors.append(
                f"{parsed.label}: generation {key} has ranks {sorted(ranks)}, "
                f"expected {sorted(expected_rank_set)}"
            )
        tx_bytes = sum(record.integer("tx_bytes") for record in records)
        rx_bytes = sum(record.integer("rx_bytes") for record in records)
        if tx_bytes != rx_bytes:
            report.errors.append(
                f"{parsed.label}: generation {key} has unbalanced bytes: "
                f"tx={tx_bytes}, rx={rx_bytes}"
            )

    perf_rank_keys = rank_keys
    if communicator == "pynccl_reference":
        if parsed.read_records:
            report.errors.append("pynccl: unexpected EPLB_READ records")
    else:
        if not parsed.read_records:
            report.errors.append(f"{parsed.label}: no EPLB_READ records")
        for record in parsed.read_records:
            missing = _missing_fields(record.values, READ_FIELDS)
            if missing:
                report.errors.append(
                    f"{parsed.label}:{record.line_number}: READ missing fields "
                    f"{sorted(missing)}"
                )
                continue
            if record.values["communicator"] != communicator:
                report.errors.append(
                    f"{parsed.label}:{record.line_number}: READ communicator "
                    "does not match the log"
                )
            try:
                if record.rank_key not in perf_rank_keys:
                    report.errors.append(
                        f"{parsed.label}:{record.line_number}: READ has no matching "
                        f"performance record {record.rank_key}"
                    )
                if record.integer("payload_bytes") <= 0:
                    report.errors.append(
                        f"{parsed.label}:{record.line_number}: payload_bytes must be "
                        "positive"
                    )
            except (KeyError, LogAnalysisError) as exc:
                report.errors.append(f"{parsed.label}: {exc}")

    if communicator == "candidate_nixl":
        if not parsed.protocol_records:
            report.errors.append("candidate: no NIXL protocol diagnostic records")
        expected_protocol_keys = {(key[3], key[1]) for key in rank_keys}
        protocol_keys: set[tuple[int, int]] = set()
        for index, record in enumerate(parsed.protocol_records, start=1):
            missing = _missing_fields(record, PROTOCOL_FIELDS)
            if missing:
                report.errors.append(
                    f"candidate: protocol record {index} is missing {sorted(missing)}"
                )
                continue
            rank = record["rank"]
            generation = record["generation"]
            if (
                isinstance(rank, bool)
                or not isinstance(rank, int)
                or isinstance(generation, bool)
                or not isinstance(generation, int)
            ):
                report.errors.append(
                    f"candidate: protocol record {index} has invalid rank/generation"
                )
                continue
            key = (rank, generation)
            if key in protocol_keys:
                report.errors.append(
                    f"candidate: duplicate protocol record for rank/generation {key}"
                )
            protocol_keys.add(key)
            if record.get("sync_protocol") is not True:
                report.errors.append(
                    f"candidate: protocol record {index} is not synchronous"
                )
            for counter in (
                "duplicate_notifications",
                "stale_notifications",
                "abort_sent",
                "abort_received",
                "abort_send_failures",
            ):
                if record.get(counter) != 0:
                    report.errors.append(
                        f"candidate: protocol record {index} has {counter}="
                        f"{record.get(counter)!r}"
                    )
        if protocol_keys != expected_protocol_keys:
            report.errors.append(
                "candidate: protocol rank/generation keys do not match performance "
                f"records; missing={sorted(expected_protocol_keys - protocol_keys)}, "
                f"extra={sorted(protocol_keys - expected_protocol_keys)}"
            )
        if parsed.protocol_records and not any(
            record.get("ready_sent", 0) or record.get("ready_received", 0)
            for record in parsed.protocol_records
        ):
            report.errors.append("candidate: no READY activity in protocol records")
        if parsed.protocol_records and not any(
            record.get("read_done_attached", 0) or record.get("read_done_received", 0)
            for record in parsed.protocol_records
        ):
            report.errors.append("candidate: no READ_DONE activity in protocol records")
    return report


def validate_client_log(
    parsed: ParsedClientLog,
    expected_phases: int,
) -> ValidationReport:
    """Validate phase completeness and request success for one client log."""
    report = ValidationReport()
    phases: dict[int, ClientPhaseRecord] = {}
    for record in parsed.phases:
        missing = _missing_fields(record.values, CLIENT_PHASE_FIELDS)
        if missing:
            report.errors.append(
                f"{parsed.label} client:{record.line_number}: missing fields "
                f"{sorted(missing)}"
            )
            continue
        try:
            phase = record.integer("phase")
            requests = record.integer("requests")
            successful = record.integer("successful_requests")
            new_rearrangements = record.integer("new_rearrangements")
            for name in (
                "phase_elapsed_s",
                "request_latency_mean_ms",
                "request_latency_median_ms",
                "request_latency_p95_ms",
                "request_latency_max_ms",
            ):
                if record.number(name) < 0:
                    report.errors.append(
                        f"{parsed.label} client:{record.line_number}: {name} is "
                        "negative"
                    )
        except (KeyError, LogAnalysisError) as exc:
            report.errors.append(f"{parsed.label} client: {exc}")
            continue
        if phase in phases:
            report.errors.append(f"{parsed.label} client: duplicate phase {phase}")
        phases[phase] = record
        if successful != requests:
            report.errors.append(
                f"{parsed.label} client: phase {phase} has "
                f"{successful}/{requests} successful requests"
            )
        if new_rearrangements < 1:
            report.errors.append(
                f"{parsed.label} client: phase {phase} observed no rearrangement"
            )

    expected = set(range(expected_phases))
    actual = set(phases)
    if actual != expected:
        report.errors.append(
            f"{parsed.label} client: phases do not match 0..{expected_phases - 1}; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return report


def validate_client_consistency(
    clients: Sequence[ParsedClientLog],
) -> ValidationReport:
    """Validate that every communicator used the same generated prompts."""
    report = ValidationReport()
    by_label = {
        client.label: {record.integer("phase"): record for record in client.phases}
        for client in clients
    }
    common_phases = set.intersection(*(set(records) for records in by_label.values()))
    for phase in sorted(common_phases):
        values = {
            label: (
                records[phase].values.get("seed"),
                records[phase].values.get("prompt_len"),
                records[phase].values.get("prompt_sha256"),
            )
            for label, records in by_label.items()
        }
        if len(set(values.values())) != 1:
            report.errors.append(
                f"client phase {phase}: prompt identity differs: {values}"
            )
    return report


def _percentile(values: Sequence[float], percent: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _stats(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": math.nan, "median": math.nan, "p95": math.nan}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": _percentile(values, 95),
    }


def summarize_generations(
    parsed: ParsedLog,
    residual_threshold_pct: float,
    negative_residual_tolerance_ms: float,
) -> list[GenerationSummary]:
    """Aggregate rank records while preserving the critical rank's phases."""
    grouped: dict[GenerationKey, list[PerfRecord]] = defaultdict(list)
    for record in parsed.perf_records:
        grouped[record.generation_key].append(record)

    summaries: list[GenerationSummary] = []
    communicator = parsed.expected_communicator
    inner_field = INNER_TIME_FIELDS[communicator]
    for key, records in sorted(grouped.items()):
        critical = max(records, key=lambda record: record.number("generation_wall_ms"))
        wall_values = [record.number("generation_wall_ms") for record in records]
        wall_mean = statistics.mean(wall_values)
        rank_cov = statistics.pstdev(wall_values) / wall_mean if wall_mean else 0.0
        tx_bytes = sum(record.integer("tx_bytes") for record in records)
        rx_bytes = sum(record.integer("rx_bytes") for record in records)

        candidate_values: dict[str, float | str | None] = {
            "ready_wait_ms": None,
            "read_execution_ms": None,
            "read_done_wait_ms": None,
            "protocol_residual_ms": None,
            "residual_ms": None,
            "residual_pct": None,
            "residual_flag": "",
        }
        if communicator == "candidate_nixl":
            ready_wait = critical.number("ready_wait_ms")
            read_execution = critical.number("read_execution_ms")
            read_done_wait = critical.number("read_done_wait_ms")
            staging = critical.number("staging_ms")
            wall = critical.number("generation_wall_ms")
            residual = wall - ready_wait - read_execution - read_done_wait - staging
            residual_pct = 100 * residual / wall if wall else math.nan
            flags = []
            if residual_pct > residual_threshold_pct:
                flags.append("HIGH_RESIDUAL")
            if residual < -negative_residual_tolerance_ms:
                flags.append("NEGATIVE_RESIDUAL")
            candidate_values = {
                "ready_wait_ms": ready_wait,
                "read_execution_ms": read_execution,
                "read_done_wait_ms": read_done_wait,
                "protocol_residual_ms": critical.number("protocol_residual_ms"),
                "residual_ms": residual,
                "residual_pct": residual_pct,
                "residual_flag": ";".join(flags),
            }

        summaries.append(
            GenerationSummary(
                communicator=communicator,
                rearrangement_id=key[0],
                generation_id=key[1],
                layer_id=key[2],
                rank_count=len(records),
                critical_rank=critical.integer("rank"),
                generation_wall_ms=critical.number("generation_wall_ms"),
                staging_ms=critical.number("staging_ms"),
                inner_time_source=inner_field,
                inner_time_ms=critical.number(inner_field),
                tx_bytes_total=tx_bytes,
                rx_bytes_total=rx_bytes,
                unique_payload_bytes=rx_bytes,
                rank_cov=rank_cov,
                bytes_balanced=tx_bytes == rx_bytes,
                **candidate_values,
            )
        )
    return summaries


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        return
    fieldnames: list[str] = []
    for row in materialized:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def _raw_perf_rows(logs: Sequence[ParsedLog]) -> list[dict[str, object]]:
    rows = []
    for parsed in logs:
        for record in parsed.perf_records:
            rows.append(
                {
                    "label": parsed.label,
                    "source_log": str(parsed.path),
                    "line_number": record.line_number,
                    **record.values,
                }
            )
    return rows


def _raw_read_rows(logs: Sequence[ParsedLog]) -> list[dict[str, object]]:
    rows = []
    for parsed in logs:
        for record in parsed.read_records:
            rows.append(
                {
                    "label": parsed.label,
                    "source_log": str(parsed.path),
                    "line_number": record.line_number,
                    **record.values,
                }
            )
    return rows


def _client_phase_rows(
    clients: Sequence[ParsedClientLog],
) -> list[dict[str, object]]:
    rows = []
    for parsed in clients:
        for record in parsed.phases:
            rows.append(
                {
                    "label": parsed.label,
                    "source_log": str(parsed.path),
                    "line_number": record.line_number,
                    **record.values,
                }
            )
    return rows


def _generation_rows(summaries: Sequence[GenerationSummary]) -> list[dict[str, object]]:
    return [asdict(summary) for summary in summaries]


def _candidate_phase_rows(
    summaries: Sequence[GenerationSummary],
) -> list[dict[str, object]]:
    candidate = [s for s in summaries if s.communicator == "candidate_nixl"]
    phases = {
        "ready_wait_ms": [s.ready_wait_ms for s in candidate],
        "read_execution_ms": [s.read_execution_ms for s in candidate],
        "read_done_wait_ms": [s.read_done_wait_ms for s in candidate],
        "protocol_residual_ms": [s.protocol_residual_ms for s in candidate],
        "staging_ms": [s.staging_ms for s in candidate],
        "residual_ms": [s.residual_ms for s in candidate],
        "residual_pct": [s.residual_pct for s in candidate],
    }
    rows = []
    for phase, values in phases.items():
        numeric = [float(value) for value in values if value is not None]
        rows.append({"phase": phase, **_stats(numeric)})
    return rows


def _comparison_rows(
    logs: Sequence[ParsedLog],
    summaries: Sequence[GenerationSummary],
    clients: Sequence[ParsedClientLog],
) -> list[dict[str, object]]:
    rows = []
    clients_by_label = {client.label: client for client in clients}
    for parsed in logs:
        communicator = parsed.expected_communicator
        selected = [s for s in summaries if s.communicator == communicator]
        outer = _stats(parsed.rearrangement_ms)
        wall = _stats([s.generation_wall_ms for s in selected])
        inner = _stats([s.inner_time_ms for s in selected])
        payload = _stats([float(s.unique_payload_bytes) for s in selected])
        cov = _stats([s.rank_cov for s in selected])
        client = clients_by_label.get(parsed.label)
        phase_elapsed = _stats(
            [record.number("phase_elapsed_s") * 1000 for record in client.phases]
            if client
            else []
        )
        request_count = (
            sum(record.integer("requests") for record in client.phases) if client else 0
        )
        weighted_request_mean = (
            sum(
                record.number("request_latency_mean_ms") * record.integer("requests")
                for record in client.phases
            )
            / request_count
            if client and request_count
            else math.nan
        )
        phase_p95 = _stats(
            [record.number("request_latency_p95_ms") for record in client.phases]
            if client
            else []
        )
        rows.append(
            {
                "communicator": communicator,
                "rearrangement_count": outer["count"],
                "rearrangement_wall_mean_ms": outer["mean"],
                "rearrangement_wall_median_ms": outer["median"],
                "rearrangement_wall_p95_ms": outer["p95"],
                "generation_count": wall["count"],
                "generation_wall_mean_ms": wall["mean"],
                "generation_wall_median_ms": wall["median"],
                "generation_wall_p95_ms": wall["p95"],
                "inner_time_source": INNER_TIME_FIELDS[communicator],
                "inner_time_mean_ms": inner["mean"],
                "inner_time_median_ms": inner["median"],
                "inner_time_p95_ms": inner["p95"],
                "unique_payload_mean_bytes": payload["mean"],
                "unique_payload_median_bytes": payload["median"],
                "unique_payload_p95_bytes": payload["p95"],
                "rank_cov_mean": cov["mean"],
                "rank_cov_median": cov["median"],
                "rank_cov_p95": cov["p95"],
                "client_phase_count": phase_elapsed["count"],
                "client_phase_elapsed_mean_ms": phase_elapsed["mean"],
                "client_phase_elapsed_median_ms": phase_elapsed["median"],
                "client_phase_elapsed_p95_ms": phase_elapsed["p95"],
                "request_count": request_count,
                "request_latency_weighted_mean_ms": weighted_request_mean,
                "request_latency_phase_p95_mean_ms": phase_p95["mean"],
                "residual_flag_count": sum(bool(s.residual_flag) for s in selected),
            }
        )
    return rows


def _comparison_delta_rows(
    comparison_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    by_communicator = {row["communicator"]: row for row in comparison_rows}
    pairs = (
        ("candidate_vs_baseline", "baseline_nixl", "candidate_nixl"),
        ("pynccl_vs_candidate", "candidate_nixl", "pynccl_reference"),
    )
    metrics = (
        "rearrangement_wall_mean_ms",
        "generation_wall_mean_ms",
        "inner_time_mean_ms",
        "unique_payload_mean_bytes",
        "rank_cov_mean",
        "client_phase_elapsed_mean_ms",
        "request_latency_weighted_mean_ms",
    )
    rows = []
    for comparison, reference_name, value_name in pairs:
        reference_row = by_communicator[reference_name]
        value_row = by_communicator[value_name]
        for metric in metrics:
            reference = float(reference_row[metric])
            value = float(value_row[metric])
            percent_change = 100 * (value / reference - 1) if reference else math.nan
            rows.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "reference": reference,
                    "value": value,
                    "percent_change": percent_change,
                }
            )
    return rows


def _format_number(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        return f"{value:.3f}"
    return str(value)


def _write_summary(
    path: Path,
    comparison_rows: Sequence[dict[str, object]],
    delta_rows: Sequence[dict[str, object]],
    phase_rows: Sequence[dict[str, object]],
    report: ValidationReport,
) -> None:
    lines = [
        "# EPLB performance analysis",
        "",
        f"Validation: **{report.status}**",
        "",
        "## Communicator comparison",
        "",
        "| Communicator | Rearrangement mean (ms) | Generation mean (ms) | "
        "Inner mean (ms) | Inner source | Payload mean (bytes) | Rank CoV mean | "
        "Client phase mean (ms) | Request mean (ms) |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison_rows:
        lines.append(
            "| {communicator} | {outer} | {wall} | {inner} | {source} | "
            "{payload} | {cov} | {phase} | {request} |".format(
                communicator=row["communicator"],
                outer=_format_number(row["rearrangement_wall_mean_ms"]),
                wall=_format_number(row["generation_wall_mean_ms"]),
                inner=_format_number(row["inner_time_mean_ms"]),
                source=row["inner_time_source"],
                payload=_format_number(row["unique_payload_mean_bytes"]),
                cov=_format_number(row["rank_cov_mean"]),
                phase=_format_number(row["client_phase_elapsed_mean_ms"]),
                request=_format_number(row["request_latency_weighted_mean_ms"]),
            )
        )
    lines.extend(
        [
            "",
            "## Mean changes",
            "",
            "Negative values mean the numerator communicator is lower/faster.",
            "",
            "| Comparison | Metric | Change |",
            "| --- | --- | ---: |",
        ]
    )
    for row in delta_rows:
        lines.append(
            f"| {row['comparison']} | {row['metric']} | "
            f"{_format_number(row['percent_change'])}% |"
        )
    lines.extend(
        [
            "",
            "## Candidate critical-rank phase summary",
            "",
            "`residual_ms` is generation wall time minus READY wait, READ "
            "execution, READ_DONE wait, and staging. It includes "
            "`protocol_residual_ms`; do not add both residual columns together.",
            "",
            "| Phase | Count | Mean | Median | P95 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in phase_rows:
        lines.append(
            "| {phase} | {count} | {mean} | {median} | {p95} |".format(
                **{name: _format_number(value) for name, value in row.items()}
            )
        )
    if report.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report.warnings)
    path.write_text("\n".join(lines) + "\n")


def analyze(config: AnalysisConfig) -> tuple[ValidationReport, list[Path]]:
    """Run validation and analysis, returning the report and generated files."""
    logs = [
        parse_log("baseline", config.baseline_log),
        parse_log("candidate", config.candidate_log),
        parse_log("pynccl", config.pynccl_log),
    ]
    client_paths = (
        config.baseline_client_log,
        config.candidate_client_log,
        config.pynccl_client_log,
    )
    if any(client_paths) and not all(client_paths):
        raise LogAnalysisError("provide either all three client logs or none")
    clients = (
        [
            parse_client_log("baseline", config.baseline_client_log),
            parse_client_log("candidate", config.candidate_client_log),
            parse_client_log("pynccl", config.pynccl_client_log),
        ]
        if all(client_paths)
        else []
    )
    report = ValidationReport()
    for parsed in logs:
        log_report = validate_log(
            parsed,
            expected_ranks=config.expected_ranks,
            min_rearrangements=config.min_rearrangements,
        )
        report.errors.extend(log_report.errors)
        report.warnings.extend(log_report.warnings)
    client_error_count = len(report.errors)
    for client in clients:
        client_report = validate_client_log(client, config.expected_phases)
        report.errors.extend(client_report.errors)
        report.warnings.extend(client_report.warnings)
    if clients and len(report.errors) == client_error_count:
        consistency_report = validate_client_consistency(clients)
        report.errors.extend(consistency_report.errors)
        report.warnings.extend(consistency_report.warnings)
    if not clients:
        report.warnings.append(
            "client logs were not supplied; prompt and client-latency analysis skipped"
        )
    counts = {parsed.label: len(parsed.rearrangement_ms) for parsed in logs}
    if len(set(counts.values())) != 1:
        report.warnings.append(f"rearrangement counts differ across logs: {counts}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = config.output_dir / "validation_report.json"
    if report.errors:
        report_path.write_text(
            json.dumps(
                {
                    "status": report.status,
                    "errors": report.errors,
                    "warnings": report.warnings,
                    "rearrangement_counts": counts,
                    "client_phase_counts": {
                        client.label: len(client.phases) for client in clients
                    },
                },
                indent=2,
            )
            + "\n"
        )
        return report, [report_path]

    summaries = [
        summary
        for parsed in logs
        for summary in summarize_generations(
            parsed,
            residual_threshold_pct=config.residual_threshold_pct,
            negative_residual_tolerance_ms=config.negative_residual_tolerance_ms,
        )
    ]
    flagged = [s for s in summaries if s.residual_flag]
    if flagged:
        report.warnings.append(
            f"{len(flagged)} candidate generations have residual flags"
        )

    perf_path = config.output_dir / "performance_records.csv"
    read_path = config.output_dir / "read_records.csv"
    client_path = config.output_dir / "client_phase_records.csv"
    generation_path = config.output_dir / "generation_summary.csv"
    phase_path = config.output_dir / "candidate_phase_summary.csv"
    comparison_path = config.output_dir / "communicator_comparison.csv"
    delta_path = config.output_dir / "comparison_deltas.csv"
    summary_path = config.output_dir / "summary.md"

    phase_rows = _candidate_phase_rows(summaries)
    comparison_rows = _comparison_rows(logs, summaries, clients)
    delta_rows = _comparison_delta_rows(comparison_rows)
    _write_csv(perf_path, _raw_perf_rows(logs))
    _write_csv(read_path, _raw_read_rows(logs))
    if clients:
        _write_csv(client_path, _client_phase_rows(clients))
    _write_csv(generation_path, _generation_rows(summaries))
    _write_csv(phase_path, phase_rows)
    _write_csv(comparison_path, comparison_rows)
    _write_csv(delta_path, delta_rows)
    _write_summary(summary_path, comparison_rows, delta_rows, phase_rows, report)

    report_path.write_text(
        json.dumps(
            {
                "status": report.status,
                "errors": report.errors,
                "warnings": report.warnings,
                "rearrangement_counts": counts,
                "performance_record_counts": {
                    parsed.label: len(parsed.perf_records) for parsed in logs
                },
                "read_record_counts": {
                    parsed.label: len(parsed.read_records) for parsed in logs
                },
                "client_phase_counts": {
                    client.label: len(client.phases) for client in clients
                },
                "residual_flag_count": len(flagged),
            },
            indent=2,
        )
        + "\n"
    )
    outputs = [
        report_path,
        perf_path,
        read_path,
        generation_path,
        phase_path,
        comparison_path,
        delta_path,
        summary_path,
    ]
    if clients:
        outputs.insert(3, client_path)
    return report, outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-log", type=Path, required=True)
    parser.add_argument("--candidate-log", type=Path, required=True)
    parser.add_argument("--pynccl-log", type=Path, required=True)
    parser.add_argument("--baseline-client-log", type=Path)
    parser.add_argument("--candidate-client-log", type=Path)
    parser.add_argument("--pynccl-client-log", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-ranks", type=int, default=8)
    parser.add_argument("--min-rearrangements", type=int, default=64)
    parser.add_argument("--expected-phases", type=int, default=64)
    parser.add_argument("--residual-threshold-pct", type=float, default=10.0)
    parser.add_argument(
        "--negative-residual-tolerance-ms",
        type=float,
        default=0.1,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = AnalysisConfig(
        baseline_log=args.baseline_log,
        candidate_log=args.candidate_log,
        pynccl_log=args.pynccl_log,
        baseline_client_log=args.baseline_client_log,
        candidate_client_log=args.candidate_client_log,
        pynccl_client_log=args.pynccl_client_log,
        output_dir=args.output_dir,
        expected_ranks=args.expected_ranks,
        min_rearrangements=args.min_rearrangements,
        expected_phases=args.expected_phases,
        residual_threshold_pct=args.residual_threshold_pct,
        negative_residual_tolerance_ms=args.negative_residual_tolerance_ms,
    )
    try:
        report, outputs = analyze(config)
    except LogAnalysisError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    for error in report.errors:
        print(f"ERROR: {error}", file=sys.stderr)
    for warning in report.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    print(f"Validation: {report.status}")
    for output in outputs:
        print(output)
    return 2 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
