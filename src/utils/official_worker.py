"""Isolated worker using the pinned, unmodified official scorer modules."""

import json
import sys
import tempfile
from pathlib import Path


def main():
    root, source, destination = map(Path, sys.argv[1:4])
    temporary = destination.parent / "scoring_tmp"
    temporary.mkdir(exist_ok=True)
    tempfile.tempdir = str(temporary)
    sys.path.insert(0, str(root))
    import measure_scores

    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    refs = [r["references"] for r in rows]
    hyps = [r["prediction"] for r in rows]
    contexts = [r.get("context", r["meaning_representation"]) for r in rows]
    coco = measure_scores.run_coco_eval(refs, hyps)
    corpus = measure_scores.run_mteval(refs, hyps, contexts)
    corpus.update(coco.eval)
    destination.write_text(
        json.dumps(dict(corpus=corpus, per_sample=coco.evalImgs)), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
