"""Summarize three distinct training seeds under one complete official test protocol."""

import argparse
import statistics
from pathlib import Path

from config import OUTPUT_ROOT, inside_project
from utils.runs import read_json, write_json


def summarize(paths):
    runs, seeds, signature = [], [], None
    for value in paths:
        path = inside_project(value)
        config = read_json(path / "config.json", {})
        metrics = read_json(path / "metrics.json", {})
        coverage = read_json(path / "coverage.json", {})
        if (
            read_json(path / "status.json", {}).get("status") != "completed"
            or metrics.get("profile") != "e2e_official"
            or metrics.get("errors")
            or not coverage.get("complete")
            or not coverage.get("full_split")
            or metrics.get("samples") != 630
            or config.get("split") != "test"
            or config.get("dataset") != "lora/e2e-original"
            or config.get("smoke")
            or config.get("local_data")
        ):
            raise ValueError(f"Not a complete original E2E test: {path}")
        spec = config["model"]
        training = spec.get("training_config")
        if not training and spec.get("training_run"):
            training = read_json(Path(spec["training_run"]) / "config.json", {})
        if (
            not training
            or training.get("dataset") != "lora/e2e-original"
            or training.get("local_data")
            or training.get("smoke")
            or training.get("train_samples") is not None
            or training.get("eval_samples") is not None
            or training.get("max_steps", -1) != -1
        ):
            raise ValueError(
                "Original-data training metadata is required; GEM weights cannot be relabelled"
            )
        seeds.append(training["seed"])
        generation = read_json(path / "generation_config.json")
        if generation is None:
            raise ValueError(
                "Use generation runs, not rescoring copies, for the three-seed summary"
            )
        current = dict(
            training={k: v for k, v in training.items() if k != "seed"},
            generation=generation,
            metrics=sorted(metrics["metrics"]),
            kind=spec["type"],
            data=read_json(path / "data_summary.json"),
            scorer=read_json(path / "scoring_source.json"),
        )
        if signature is not None and signature != current:
            raise ValueError("Training/data/generation/scoring configurations differ beyond seed")
        signature = current
        runs.append(dict(path=str(path), seed=seeds[-1], metrics=metrics["metrics"]))
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("Exactly three distinct training seeds are required")
    return dict(
        runs=runs,
        protocol=signature,
        summary={
            key: dict(
                mean=statistics.mean(r["metrics"][key] for r in runs),
                sample_sd=statistics.stdev(r["metrics"][key] for r in runs),
            )
            for key in runs[0]["metrics"]
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs=3)
    parser.add_argument("--output", default=str(OUTPUT_ROOT / "paper_three_seeds.json"))
    args = parser.parse_args()
    target = inside_project(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, summarize(args.runs))
    print(target)
