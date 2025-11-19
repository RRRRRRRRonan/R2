from __future__ import annotations

import json
import logging
import pathlib
from typing import List

# matplotlib is optional; plotting gracefully degrades when unavailable.
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    plt = None

from src.config import ExperimentConfig, load_config, parse_args
from src.data import generate_random_instance
from src.drl_agent import DRLAgent
from src.hea import HEA


def setup_logger(output_dir: pathlib.Path, level: str) -> logging.Logger:
    logger = logging.getLogger("hea")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def maybe_load_agent(config: ExperimentConfig) -> DRLAgent | None:
    if config.algorithm.removal_strategy.lower() != "drl":
        return None
    if not config.algorithm.model_path:
        raise ValueError("DRL strategy requires model_path")
    model_path = pathlib.Path(config.algorithm.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return DRLAgent.load(model_path)


def save_convergence_plot(stats: List, output_dir: pathlib.Path, label: str) -> None:
    if plt is None:
        return
    generations = [s.generation for s in stats]
    best_costs = [s.best_cost for s in stats]
    plt.figure(figsize=(6, 4))
    plt.plot(generations, best_costs, label=label)
    plt.xlabel("Generation")
    plt.ylabel("Best cost")
    plt.title("Convergence curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "convergence.png")
    plt.close()


def save_summary(result, config: ExperimentConfig, output_dir: pathlib.Path) -> None:
    summary = {
        "best_cost": result.best_solution.cost(),
        "generations": len(result.stats),
        "timing": result.timing,
        "config": {
            "problem": config.problem.__dict__,
            "algorithm": config.algorithm.__dict__,
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    instance = generate_random_instance(config.problem)
    output_dir = pathlib.Path(config.logging.output_dir)
    logger = setup_logger(output_dir, config.logging.log_level)
    logger.info("Loaded configuration: %s", config)
    agent = maybe_load_agent(config)
    algo = HEA(instance, config.algorithm, logger, agent)
    result = algo.run()
    save_convergence_plot(result.stats, output_dir, label=config.algorithm.removal_strategy)
    save_summary(result, config, output_dir)
    if config.logging.save_solution:
        result.best_solution.save(str(output_dir / "best_solution.json"))
    logger.info("Best solution cost: %.3f", result.best_solution.cost())
    logger.info("Timing: %s", result.timing)


if __name__ == "__main__":
    main()
