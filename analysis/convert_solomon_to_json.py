#!/usr/bin/env python3
"""Convert Solomon/SINTEF VRPTW benchmark files to the project's JSON schema."""

from __future__ import annotations

import argparse
import json
import pathlib


def parse_solomon_file(path: pathlib.Path) -> dict:
    raw_lines = [line.rstrip() for line in path.read_text().splitlines()]
    lines = [line for line in raw_lines if line.strip()]
    if not lines:
        raise ValueError(f"{path} is empty")
    name = lines[0].strip()
    # Find header line describing the columns
    header_idx = None
    for idx, line in enumerate(lines):
        if line.upper().startswith("CUST NO"):
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError("Unable to locate 'CUST NO.' header line in the file")
    data_lines = lines[header_idx + 1 :]
    tasks = []
    vehicle_capacity = None
    for line in data_lines:
        parts = line.split()
        if len(parts) < 7:
            continue
        cust_id = int(float(parts[0]))
        x = float(parts[1])
        y = float(parts[2])
        demand = float(parts[3])
        ready = float(parts[4])
        due = float(parts[5])
        service = float(parts[6])
        if cust_id == 0:
            vehicle_capacity = vehicle_capacity or float(parts[-1])
            depot = (x, y)
            continue
        tasks.append(
            {
                "idx": cust_id - 1,
                "x": x,
                "y": y,
                "demand": demand,
                "service_time": service,
                "ready_time": ready,
                "due_time": due,
            }
        )
    if vehicle_capacity is None:
        raise ValueError("Vehicle capacity not found in the file")
    return {
        "name": name,
        "depot": list(depot),
        "vehicle_capacity": vehicle_capacity,
        "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a Solomon/SINTEF VRPTW benchmark file to JSON."
    )
    parser.add_argument("--input", type=pathlib.Path, required=True, help="Path to R*.txt file")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="Destination JSON path (will be overwritten)",
    )
    args = parser.parse_args()
    data = parse_solomon_file(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
