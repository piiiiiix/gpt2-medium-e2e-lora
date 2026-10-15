"""评分模块：沿用 small 指标口径，保存汇总和支持的逐样本分数及失败详情。"""

import csv
import json
import tempfile
import traceback
import urllib.request
import zipfile
from pathlib import Path

from config import METRICS, OUTPUT_ROOT
from utils.runs import write_json


def read_predictions(path):
    """读取已保存预测 JSONL，拒绝对空文件评分。"""
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise ValueError("No predictions to score")
    return records


def wordnet():
    """准备并验证项目 WordNet，直接下载官方 ZIP，兼容系统代理。"""
    import nltk
    from nltk.corpus import wordnet

    folder = OUTPUT_ROOT / "resources/nltk_data"
    corpora = folder / "corpora"
    corpora.mkdir(parents=True, exist_ok=True)
    target = corpora / "wordnet.zip"
    if not target.exists() and not (corpora / "wordnet").is_dir():
        url = (
            "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip"
        )
        print(f"Downloading WordNet: {url}\nDestination: {target}", flush=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=corpora, suffix=".part", delete=False) as stream:
                temporary = Path(stream.name)
                with urllib.request.urlopen(url, timeout=60) as response:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
            check_wordnet_archive(temporary)
            temporary.replace(target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    elif target.exists():
        check_wordnet_archive(target)
    # 加载时仅查项目资源；随后仍保留项目路径供延迟读取与安全校验使用。
    previous_paths = nltk.data.path[:]
    nltk.data.path[:] = [str(folder)]
    try:
        wordnet.ensure_loaded()
        if not wordnet.synsets("restaurant"):
            raise RuntimeError("Project WordNet query returned no synsets")
    finally:
        nltk.data.path[:] = [str(folder)] + [p for p in previous_paths if p != str(folder)]


def check_wordnet_archive(path):
    """检查 ZIP CRC 和必需文件，避免将不完整下载作为可用语料。"""
    with zipfile.ZipFile(path) as archive:
        required = {f"wordnet/{name}" for name in ("lexnames", "index.noun", "data.noun")}
        if not required.issubset(archive.namelist()):
            raise ValueError(
                f"Missing WordNet files in {path}; remove the invalid archive and retry"
            )
        bad = archive.testzip()
        if bad:
            raise ValueError(f"Corrupt WordNet ZIP member: {bad}; remove the archive and retry")


def score(path, selected, output):
    """逐项计算指标并保存明细；部分指标失败时仍保留成功成绩与完整输入。"""
    rows = read_predictions(path)
    if not selected or set(selected) - set(METRICS):
        raise ValueError("Select one or more supported metrics")
    if any(row.get("error") for row in rows):
        raise ValueError("Predictions contain generation errors; inspect records before rescoring")
    gts = {i: [row["target"]] for i, row in enumerate(rows)}
    res = {i: [row["prediction"]] for i, row in enumerate(rows)}
    summary, errors, scopes = {}, {}, {}
    per_row = [dict(id=row["id"], scores={}) for row in rows]
    for metric in selected:
        try:
            if metric == "BLEU":
                from pycocoevalcap.bleu.bleu import Bleu

                corpus, individual = Bleu(4).compute_score(gts, res)
                values = {f"BLEU-{k + 1}": float(v) for k, v in enumerate(corpus)}
                items = [
                    {f"BLEU-{k + 1}": float(individual[k][i]) for k in range(4)}
                    for i in range(len(rows))
                ]
                scope = (
                    "corpus BLEU; sentence BLEU separately retained, not averaged into corpus score"
                )
            elif metric in ("CIDEr", "ROUGE-L"):
                if metric == "CIDEr":
                    from pycocoevalcap.cider.cider import Cider

                    scorer = Cider()
                else:
                    from pycocoevalcap.rouge.rouge import Rouge

                    scorer = Rouge()
                corpus, individual = scorer.compute_score(gts, res)
                values = {metric: float(corpus)}
                items = [{metric: float(v)} for v in individual]
                scope = "sample average; CIDEr uses this whole corpus document frequencies"
            elif metric == "NIST":
                from nltk.translate.nist_score import corpus_nist

                values = {
                    metric: float(
                        corpus_nist(
                            [[r["target"].split()] for r in rows],
                            [r["prediction"].split() for r in rows],
                            n=5,
                        )
                    )
                }
                items = [{metric: None} for _ in rows]
                scope = "corpus NIST-5 only; per-sample score unavailable in this scoring profile"
            else:
                wordnet()
                from nltk.translate.meteor_score import meteor_score

                individual = [
                    float(meteor_score([r["target"].split()], r["prediction"].split()))
                    for r in rows
                ]
                values = {metric: sum(individual) / len(individual)}
                items = [{metric: v} for v in individual]
                scope = "mean sample METEOR, NLTK tokenized whitespace with WordNet"
            summary.update(values)
            scopes[metric] = scope
            for target, item in zip(per_row, items):
                target["scores"].update(item)
            write_json(
                output / f"{metric}_scores.json",
                dict(summary=values, scope=scope, per_sample=items),
            )
        except Exception as exc:
            errors[metric] = dict(error=str(exc), traceback=traceback.format_exc())
            write_json(output / f"{metric}_error.json", errors[metric])
        write_json(
            output / "metrics.json",
            dict(
                metrics=summary,
                errors=errors,
                scope=scopes,
                samples=len(rows),
                selected=selected,
                profile="small-compatible single-target",
            ),
        )
        with (output / "sample_scores.jsonl").open("w", encoding="utf-8") as stream:
            for row, result in zip(rows, per_row):
                stream.write(json.dumps({**row, **result}, ensure_ascii=False) + "\n")
    with (output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["metric", "score"])
        writer.writeheader()
        writer.writerows(dict(metric=k, score=v) for k, v in summary.items())
    if errors:
        raise RuntimeError(
            f"Some metrics failed; successful scores and full predictions preserved: {list(errors)}"
        )
    return summary
