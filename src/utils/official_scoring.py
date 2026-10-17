"""Official E2E: Perl MTEval BLEU/NIST and PTB + Java METEOR/ROUGE/CIDEr."""

import csv
import json
import subprocess
import sys
from pathlib import Path

from config import METRICS
from utils.paper_resources import evaluator, manifest, runtime_env
from utils.runs import write_json
from utils.scoring import read_predictions


def score_official(path, selected, output, offline=False):
    if not selected or set(selected) - set(METRICS):
        raise ValueError("Select one or more supported metrics")
    rows = read_predictions(path)
    if any(
        r.get("error") or not isinstance(r.get("prediction"), str) or not r.get("references")
        for r in rows
    ):
        raise ValueError("Official scoring requires valid predictions and all references")
    root = evaluator(offline)
    env = runtime_env()
    # Human-reviewable files follow gpt2_decode.py's original multi-reference format.
    (output / "references.txt").write_text(
        "".join("\n".join(r["references"]) + "\n\n" for r in rows), encoding="utf-8"
    )
    (output / "hypotheses.txt").write_text(
        "".join(r["prediction"] + "\n" for r in rows), encoding="utf-8"
    )
    raw = output / "official_scores.json"
    command = [
        sys.executable,
        str(Path(__file__).with_name("official_worker.py")),
        str(root),
        str(Path(path).resolve()),
        str(raw),
    ]
    with (output / "official_scoring.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800
        )
    result = json.loads(raw.read_text(encoding="utf-8"))
    names = {"BLEU": "BLEU-4", "ROUGE_L": "ROUGE-L"}
    wanted = {"BLEU-4" if m == "BLEU" else m for m in selected}
    summary = {names.get(k, k): v for k, v in result["corpus"].items() if names.get(k, k) in wanted}
    provenance = manifest()["evaluator"]
    write_json(
        output / "scoring_source.json", {k: provenance[k] for k in ("url", "commit", "sha256")}
    )
    write_json(
        output / "metrics.json",
        dict(
            metrics=summary,
            errors={},
            samples=len(rows),
            selected=selected,
            profile="e2e_official",
            scope={
                "BLEU": "corpus MTEval v13a-sig (Perl), all references, fraction 0..1",
                "NIST": "corpus MTEval v13a-sig (Perl), all references",
                "METEOR": "official Java METEOR 1.5 corpus score after PTB",
                "ROUGE-L": "official COCO after PTB, all references",
                "CIDEr": "official COCO after PTB, corpus document frequencies",
            },
        ),
    )
    by_id = {item["image_id"]: item for item in result["per_sample"]}
    with (output / "sample_scores.jsonl").open("w", encoding="utf-8") as stream:
        for i, row in enumerate(rows):
            scores = {
                names.get(k, k): v
                for k, v in by_id[f"inst-{i}"].items()
                if names.get(k, k) in wanted
            }
            stream.write(json.dumps({**row, "scores": scores}, ensure_ascii=False) + "\n")
    with (output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "score"])
        writer.writerows(summary.items())
    return summary
