#!/usr/bin/env python3
"""Run transfer experiments for predefined scenarios and save per-scenario outputs."""

from __future__ import annotations

import os
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt

def _ensure_matplotlib_dir() -> None:
    if os.environ.get("MPLCONFIGDIR"):
        return
    cache_dir = pathlib.Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache_dir)
    except OSError:
        return


_ensure_matplotlib_dir()

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.config import AlgorithmConfig, ProblemConfig  # noqa: E402
from src.data import generate_random_instance  # noqa: E402
from src.drl_agent import DRLAgent  # noqa: E402
from src.hea import HEA  # noqa: E402

DRL_MODEL = pathlib.Path("models/drl_model_transformer_backup.json")


@dataclass
class TransferRecord:
    problem: str
    method: str
    tardiness: float
    time_sec: float


SCENARIOS = [
    ("Solomon_C101", pathlib.Path("data/transfer_scenarios/solomon_c101_sample.json")),
    ("Homberger_H1", pathlib.Path("data/transfer_scenarios/homberger_h1_sample.json")),
    ("LiLim_EVRPTW", pathlib.Path("data/transfer_scenarios/lilim_evrptw_sample.json")),
    ("Uchoa_CVRPTW_PDPTW", pathlib.Path("data/transfer_scenarios/uchoa_cvrptw_pdptw_sample.json")),
]


def run_scenario(name: str, path: pathlib.Path, seeds: List[int]) -> List[TransferRecord]:
    methods = [
        ("HEA-DRL", "drl_hybrid"),
        ("HEA-Shaw", "shaw"),
        ("HEA-Random", "random"),
    ]
    records: List[TransferRecord] = []
    cfg = ProblemConfig(data_file=str(path))
    tmp_instance = generate_random_instance(cfg)
    remove_count = max(1, int(0.15 * len(tmp_instance.tasks)))
    for seed in seeds:
        instance = generate_random_instance(cfg)
        for method_name, strategy in methods:
            algo_cfg = AlgorithmConfig(
                removal_strategy=strategy,
                remove_count=remove_count,
                population_size=40,
                generations=60,
                crossover_rate=0.85,
                mutation_rate=0.2,
                elite_rate=0.1,
                model_path=str(DRL_MODEL) if "drl" in strategy else None,
                seed=seed,
            )
            agent = DRLAgent.load(DRL_MODEL) if "drl" in strategy else None
            start = time.perf_counter()
            algo = HEA(instance, algo_cfg, logger=None, agent=agent)  # type: ignore[arg-type]
            result = algo.run()
            elapsed = time.perf_counter() - start
            records.append(
                TransferRecord(
                    problem=name,
                    method=method_name,
                    tardiness=result.best_solution.cost(),
                    time_sec=result.timing.get("T_total", elapsed),
                )
            )
    return records


def write_summary(records: List[TransferRecord], out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# 策略迁移实验", ""]
    lines.append("| 问题 | 策略 | 总延迟 | 运行时间 (s) |")
    lines.append("| --- | --- | --- | --- |")
    for rec in records:
        lines.append(
            f"| {rec.problem} | {rec.method} | {rec.tardiness:.2f} | {rec.time_sec:.2f} |"
        )
    lines.append("")
    (out_dir / "transfer_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import json

    data = [rec.__dict__ for rec in records]
    (out_dir / "transfer_summary.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # Plot grouped bars for tardiness and runtime
    problems = list(dict.fromkeys(rec.problem for rec in records))
    methods = list(dict.fromkeys(rec.method for rec in records))
    tardiness = {
        (rec.problem, rec.method): rec.tardiness for rec in records
    }
    runtimes = {(rec.problem, rec.method): rec.time_sec for rec in records}
    positions = range(len(problems))
    width = 0.8 / max(1, len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for axis, metric_name, data in zip(
        axes,
        ["Total tardiness", "Runtime (s)"],
        [tardiness, runtimes],
    ):
        for idx, method in enumerate(methods):
            offsets = [p - 0.4 + width / 2 + idx * width for p in positions]
            axis.bar(
                offsets,
                [data.get((problem, method), 0.0) for problem in problems],
                width=width,
                label=method,
            )
        axis.set_xticks(list(positions))
        axis.set_xticklabels(problems, rotation=20)
        axis.set_title(metric_name)
    axes[0].set_ylabel("Value")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.0, 1.1), ncol=len(methods))
    fig.suptitle("Transfer performance comparison")
    fig.tight_layout()
    plt.savefig(out_dir / "transfer_tardiness.png")
    plt.close(fig)


def main() -> None:
    seeds = [11, 13, 17, 19]
    base_output = pathlib.Path("results/transfer")
    for name, path in SCENARIOS:
        if not path.exists():
            print(f"[warn] skip {name}, missing {path}")
            continue
        out_dir = base_output / name
        recs = run_scenario(name, path, seeds)
        write_summary(recs, out_dir)
        print(f"[done] {name} -> {out_dir}")


if __name__ == "__main__":
    main()
