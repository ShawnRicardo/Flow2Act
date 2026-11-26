#!/usr/bin/env python3
"""Summarize completed LIBERO/LIBERO-Plus rollouts without simulator imports.

Rates use successful / completed episodes. Technical errors never become policy
failures. The four-suite macro average gives each suite equal weight; the Plus
category rates are episode-weighted within each official perturbation category.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
BENCHMARKS = ("libero", "libero_plus")
DEFAULT_CLASSIFICATION = (
    Path(os.environ.get("LIBERO_PLUS_ROOT", Path(__file__).resolve().parents[1] / "third_party/LIBERO-plus"))
    / "libero/libero/benchmark/task_classification.json"
)


def load_catalog(path: Path) -> dict[str, list[dict[str, Any]]]:
    catalog = json.loads(path.read_text(encoding="utf-8"))
    if set(catalog) != set(SUITES):
        raise ValueError(f"Expected four LIBERO suites in classification: {path}")
    for suite, rows in catalog.items():
        if not rows or len({row["name"] for row in rows}) != len(rows):
            raise ValueError(f"Empty or duplicate task names in {suite}")
        for row in rows:
            if not isinstance(row.get("category"), str):
                raise ValueError(f"Missing perturbation category in {suite}: {row}")
    return catalog


def metric(expected: int, records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in records if row["status"] == "completed"]
    successes = sum(row["success"] for row in completed)
    return {
        "completed": len(completed),
        "expected": expected,
        "successes": successes,
        "failures": len(completed) - successes,
        "technical_errors": sum(row["status"] == "error" for row in records),
        "success_rate": successes / len(completed) if completed else None,
    }


def build_snapshot(output_dir: Path, classification: Path = DEFAULT_CLASSIFICATION, seed: int = 7, *, benchmarks=BENCHMARKS, suites=SUITES, libero_trials=50) -> dict[str, Any]:
    benchmarks, suites = tuple(benchmarks), tuple(suites)
    if not benchmarks or len(set(benchmarks)) != len(benchmarks) or any(x not in BENCHMARKS for x in benchmarks):
        raise ValueError("Invalid benchmark selection")
    if not suites or len(set(suites)) != len(suites) or any(x not in SUITES for x in suites):
        raise ValueError("Invalid suite selection")
    selected_suites = suites
    catalog = load_catalog(classification) if "libero_plus" in benchmarks else {name: [] for name in SUITES}
    categories = sorted({row["category"] for rows in catalog.values() for row in rows})
    expected_categories = {
        suite: Counter(row["category"] for row in catalog[suite]) for suite in SUITES
    }
    names = {suite: {row["name"]: row for row in catalog[suite]} for suite in SUITES}
    candidates: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    data_errors: list[dict[str, Any]] = []
    ignored_partial_lines: list[str] = []
    ignored_smoke_records = 0
    files_read = 0

    for benchmark in benchmarks:
        for path in sorted((output_dir / benchmark).glob("shard-*/**/episodes.jsonl")):
            files_read += 1
            raw_lines = path.read_bytes().splitlines(keepends=True)
            for line_no, raw in enumerate(raw_lines, 1):
                if not raw.strip():
                    continue
                location = f"{path.relative_to(output_dir)}:{line_no}"
                try:
                    row = json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeDecodeError) as error:
                    # A writer can be halfway through its last JSON record. A
                    # newline-terminated malformed row is an actual data error.
                    if line_no == len(raw_lines) and not raw.endswith(b"\n"):
                        ignored_partial_lines.append(location)
                    else:
                        data_errors.append({"source": location, "error": str(error)})
                    continue
                try:
                    if not isinstance(row, dict):
                        raise ValueError("Episode record must be a JSON object")
                    if row.get("smoke_only", False):
                        ignored_smoke_records += 1
                        continue
                    family = row.get("benchmark", row.get("family", benchmark))
                    if family != benchmark:
                        raise ValueError(f"Record benchmark {family!r} differs from directory {benchmark!r}")
                    suite = row["suite"]
                    if suite not in selected_suites:
                        raise ValueError(f"Unexpected suite: {suite!r}")
                    task_id, episode_idx = row["task_id"], row["episode_idx"]
                    if type(task_id) is not int or type(episode_idx) is not int:
                        raise ValueError("task_id and episode_idx must be integer indices")
                    task_count = 10 if benchmark == "libero" else len(catalog[suite])
                    episode_count = libero_trials if benchmark == "libero" else 1
                    if not 0 <= task_id < task_count or not 0 <= episode_idx < episode_count:
                        raise ValueError(f"Out-of-range task/episode index: {task_id}/{episode_idx}")
                    if row.get("seed", seed) != seed:
                        raise ValueError(f"Expected seed {seed}; found {row['seed']!r}")
                    status = row["status"]
                    if status not in ("completed", "error"):
                        raise ValueError(f"Unexpected episode status: {status!r}")
                    if status == "completed" and type(row.get("success")) is not bool:
                        raise ValueError("Completed episodes must have boolean success")
                    if benchmark == "libero_plus":
                        category = row.get("perturbation", row.get("category"))
                        if category not in expected_categories[suite]:
                            raise ValueError(f"Unknown perturbation category: {category!r}")
                        task_name = row.get("task_name")
                        if task_name is not None:
                            if task_name not in names[suite]:
                                raise ValueError(f"Unknown Plus task name: {task_name!r}")
                            if names[suite][task_name]["category"] != category:
                                raise ValueError(f"Task/category mismatch: {task_name!r}/{category!r}")
                        row = {**row, "perturbation": category}
                    candidates[(benchmark, suite, task_id, episode_idx)].append({**row, "source": location})
                except (KeyError, TypeError, ValueError) as error:
                    data_errors.append({"source": location, "error": str(error)})

    accepted: dict[str, list[dict[str, Any]]] = {key: [] for key in benchmarks}
    duplicate_episodes = []
    for identity, rows in candidates.items():
        if len(rows) != 1:
            # Reject the entire ambiguous identity; silently keeping the first
            # could inflate or depress success after an accidental duplicate run.
            duplicate_episodes.append({"identity": list(identity), "sources": [r["source"] for r in rows]})
        else:
            accepted[identity[0]].append(rows[0])

    results: dict[str, Any] = {}
    for benchmark in benchmarks:
        records = accepted[benchmark]
        suites = {}
        for suite in selected_suites:
            suite_records = [row for row in records if row["suite"] == suite]
            expected = 10 * libero_trials if benchmark == "libero" else len(catalog[suite])
            suites[suite] = metric(expected, suite_records)
            if benchmark == "libero_plus":
                suites[suite]["perturbations"] = {
                    category: metric(
                        expected_categories[suite][category],
                        [row for row in suite_records if row["perturbation"] == category],
                    )
                    for category in categories
                }
        rates = [row["success_rate"] for row in suites.values() if row["success_rate"] is not None]
        total = metric(sum(row["expected"] for row in suites.values()), records)
        total.update(
            suites=suites,
            macro_success_rate=sum(rates) / len(selected_suites) if len(rates) == len(selected_suites) else None,
            observed_suite_macro_rate=sum(rates) / len(rates) if rates else None,
            suites_with_results=len(rates),
        )
        if benchmark == "libero_plus":
            total["perturbations"] = {
                category: metric(
                    sum(expected_categories[suite][category] for suite in selected_suites),
                    [row for row in records if row["perturbation"] == category],
                )
                for category in categories
            }
            for category, row in total["perturbations"].items():
                if row["completed"] > row["expected"]:
                    data_errors.append({"source": "aggregate", "error": f"Too many completed {category}: {row['completed']}/{row['expected']}"})
        results[benchmark] = total

    technical_errors = sum(row["technical_errors"] for row in results.values())
    complete = (
        all(row["completed"] == row["expected"] for row in results.values())
        and not technical_errors
        and not data_errors
        and not duplicate_episodes
        and not ignored_partial_lines
    )
    return {
        "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "output_dir": str(output_dir.resolve()),
        "seed": seed,
        "selected_suites": list(selected_suites),
        "complete": complete,
        "metric_definition": {
            "success_rate": "successes / completed; technical errors excluded",
            "macro_success_rate": "equal mean of selected suite success rates; null until all selected suites have results",
            "perturbation_success_rate": "episode-weighted success within each official category",
        },
        "benchmarks": results,
        "technical_errors": technical_errors,
        "data_errors": data_errors,
        "duplicate_episodes": duplicate_episodes,
        "ignored_partial_lines": ignored_partial_lines,
        "ignored_smoke_records": ignored_smoke_records,
        "files_read": files_read,
    }


def rate(value: float | None) -> str:
    return "pending" if value is None else f"{value:.2%}"


def format_snapshot(snapshot: dict[str, Any]) -> str:
    lines = [
        f"Updated {snapshot['updated_at']} | seed={snapshot['seed']} | complete={snapshot['complete']}",
        "Success rate = successful/completed; technical errors excluded. Macro = equal mean of selected suites.",
    ]
    for benchmark, result in snapshot["benchmarks"].items():
        lines.append(f"\n{benchmark.upper()}  completed {result['completed']}/{result['expected']}  success {rate(result['success_rate'])}  technical_errors={result['technical_errors']}")
        for suite, row in result["suites"].items():
            lines.append(f"  {suite:16s}  {row['completed']:5d}/{row['expected']:<5d}  success {rate(row['success_rate']):>8s}  ({row['successes']} successes)  errors={row['technical_errors']}")
        lines.append(f"  Selected-suite macro success: {rate(result['macro_success_rate'])} ({result['suites_with_results']}/{len(snapshot['selected_suites'])} suites have completed episodes)")
        if result["macro_success_rate"] is None and result["suites_with_results"]:
            lines.append(f"  Observed suites only (provisional): {rate(result['observed_suite_macro_rate'])}")
        for category, row in result.get("perturbations", {}).items():
            lines.append(f"  {category:23s}  {row['completed']:5d}/{row['expected']:<5d}  success {rate(row['success_rate']):>8s}  errors={row['technical_errors']}")
    lines.append(f"\nData errors={len(snapshot['data_errors'])}; duplicate episode IDs={len(snapshot['duplicate_episodes'])}; incomplete trailing JSON lines={len(snapshot['ignored_partial_lines'])}; smoke records excluded={snapshot['ignored_smoke_records']}")
    if snapshot["data_errors"] or snapshot["duplicate_episodes"]:
        lines.append("Inspect progress.json for data error and duplicate details; ambiguous duplicates are excluded.")
    return "\n".join(lines) + "\n"


def write_snapshot(output_dir: Path, snapshot: dict[str, Any], final: bool = False) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    text = format_snapshot(snapshot)
    files = {"progress.json": json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", "progress.txt": text}
    if final and snapshot["complete"]:
        files["summary.json"] = files["progress.json"]
    for name, content in files.items():
        destination = output_dir / name
        temporary = output_dir / f".{name}.{os.getpid()}.tmp"
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(destination)
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--classification", type=Path, default=DEFAULT_CLASSIFICATION)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--benchmarks", default=",".join(BENCHMARKS))
    parser.add_argument("--suites", default=",".join(SUITES))
    parser.add_argument("--libero-trials", type=int, default=50)
    parser.add_argument("--once", action="store_true", help="One snapshot (the default; no background loop)")
    parser.add_argument("--final", action="store_true", help="Also write summary.json after all episodes complete without errors")
    args = parser.parse_args()
    snapshot = build_snapshot(args.output_dir, args.classification, args.seed, benchmarks=args.benchmarks.split(","), suites=args.suites.split(","), libero_trials=args.libero_trials)
    print(write_snapshot(args.output_dir, snapshot, args.final), end="", flush=True)
    if snapshot["data_errors"] or snapshot["duplicate_episodes"]:
        return 1
    return 2 if args.final and not snapshot["complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
