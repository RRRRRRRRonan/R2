from __future__ import annotations

import argparse
import dataclasses
import pathlib
from typing import Any, Dict

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - the lightweight parser will be used
    yaml = None


@dataclasses.dataclass
class ProblemConfig:
    instance_type: str = "random"
    num_tasks: int = 50
    distribution: str = "uniform"
    time_windows: bool = False
    seed: int | None = None
    data_file: str | None = None


@dataclasses.dataclass
class AlgorithmConfig:
    removal_strategy: str = "random"
    remove_count: int = 5
    population_size: int = 30
    generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    elite_rate: float = 0.1
    model_path: str | None = None
    use_gpu: bool = False
    train_mode: bool = False
    seed: int | None = None


@dataclasses.dataclass
class LoggingConfig:
    log_level: str = "INFO"
    output_dir: str = "results/default"
    save_solution: bool = True


@dataclasses.dataclass
class TrainingConfig:
    iterations: int = 200
    learning_rate: float = 0.05
    noise_scale: float = 0.1
    validation_interval: int = 20


@dataclasses.dataclass
class ExperimentConfig:
    problem: ProblemConfig = dataclasses.field(default_factory=ProblemConfig)
    algorithm: AlgorithmConfig = dataclasses.field(default_factory=AlgorithmConfig)
    logging: LoggingConfig = dataclasses.field(default_factory=LoggingConfig)
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)


def _simple_yaml(text: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack: list[tuple[int, Dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip() or line.strip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        key, _, value = line.lstrip().partition(":")
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1] if stack else root
        if not value:
            new_map: Dict[str, Any] = {}
            current[key] = new_map
            stack.append((indent, new_map))
        else:
            current[key] = _coerce_value(value)
    return root


def _coerce_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value


def _load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)
    return _simple_yaml(text)


def load_config(path: pathlib.Path) -> ExperimentConfig:
    raw = _load_yaml(path)
    problem = ProblemConfig(**raw.get("problem", {}))
    algorithm = AlgorithmConfig(**raw.get("algorithm", {}))
    logging_cfg = LoggingConfig(**raw.get("logging", {}))
    training = TrainingConfig(**raw.get("training", {}))
    return ExperimentConfig(problem=problem, algorithm=algorithm, logging=logging_cfg, training=training)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HEA-DRL experiments")
    parser.add_argument("--config", type=pathlib.Path, required=True, help="Path to YAML config")
    parser.add_argument("--train", action="store_true", help="Run DRL training")
    return parser.parse_args()
