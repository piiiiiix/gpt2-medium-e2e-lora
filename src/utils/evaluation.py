"""生成与评测后端：逐条保存预测，再计算指标；重新评分时不加载模型。"""

import json
import shutil
import time

import torch
from transformers import set_seed

from config import inside_project
from utils.data import evaluation_data, prompt
from utils.models import load_model
from utils.protocols import clean_prediction, resolve_generation
from utils.runs import running, write_json
from utils.scoring import read_predictions, score


def check_coverage(rows, dataset):
    """Reject partial, reordered or foreign predictions even if a scoring task succeeded."""
    if len(rows) != len(dataset):
        raise ValueError(
            f"Incomplete predictions: {len(rows)}/{len(dataset)}; rerun generation, not rescoring"
        )
    for i, (row, sample) in enumerate(zip(rows, dataset)):
        if (
            row.get("index") != i
            or row.get("meaning_representation") != sample["meaning_representation"]
        ):
            raise ValueError(f"Prediction input/order mismatch at {i}")
        refs = sample.get("references") or [sample["target"]]
        if row.get("target") != sample["target"] or row.get("references") != refs:
            raise ValueError(f"Prediction reference mismatch at {i}")
        if row.get("error") or not isinstance(row.get("prediction"), str):
            raise ValueError(f"Invalid prediction at {i}")


def evaluate(config, path, kind="benchmarks"):
    """执行生成或读取已保存预测，持久保存每条结果后运行所选指标。"""
    with running(path, kind, config):
        started = time.perf_counter()
        dataset = evaluation_data(config)
        write_json(
            path / "data_summary.json",
            dict(
                dataset=config["dataset"],
                local_data=config["local_data"],
                split=config["split"],
                count=len(dataset),
                fingerprint=dataset._fingerprint,
            ),
        )
        if config.get("scoring_profile", "legacy_single") == "e2e_official":
            from utils.paper_resources import manifest, preflight

            preflight(config["local_files_only"])
            write_json(path / "paper_sources.json", manifest())
        if config.get("predictions"):
            source = inside_project(config["predictions"])
            check_coverage(read_predictions(source), dataset)
            shutil.copyfile(source, path / "predictions.jsonl")
            write_json(
                path / "prediction_source.json",
                dict(source=str(source), source_config=str(source.parent / "config.json")),
            )
        else:
            generation = resolve_generation(config)
            generation["engine"] = (
                "lora_source_beam"
                if generation["profile"] in ("lora_paper", "lora_readme")
                else "transformers_generate"
            )
            write_json(path / "generation_config.json", generation)
            set_seed(config["seed"])
            write_json(
                path / "data_summary.json",
                dict(
                    dataset=config["dataset"],
                    local_data=config["local_data"],
                    split=config["split"],
                    count=len(dataset),
                    fingerprint=dataset._fingerprint,
                ),
            )
            model, tokenizer = load_model(config["model"], config["local_files_only"])
            generation["eos_token_id"] = generation["eos_token_id"] or tokenizer.eos_token_id
            generation["pad_token_id"] = tokenizer.pad_token_id
            generation["bad_words_ids"] = (
                [[628], [198]] if generation["protocol"] == "prefix_ft" else None
            )
            generation["repetition_penalty"] = 1.0
            write_json(path / "generation_config.json", generation)
            model.eval()
            model.config.use_cache = True
            generation_errors = 0
            try:
                with (path / "predictions.jsonl").open(
                    "x", encoding="utf-8", buffering=1
                ) as stream:
                    for i, sample in enumerate(dataset):
                        record = dict(
                            id=sample.get("gem_id", sample.get("id", i)),
                            index=i,
                            meaning_representation=sample["meaning_representation"],
                            prompt=prompt(sample, generation["protocol"], tokenizer),
                            target=sample["target"],
                            references=sample.get("references") or [sample["target"]],
                        )
                        if "context" in sample:
                            record["context"] = sample["context"]
                        sample_started = time.perf_counter()
                        try:
                            inputs = tokenizer(record["prompt"], return_tensors="pt").to(
                                model.device
                            )
                            length = inputs["input_ids"].shape[1]
                            if length + generation["max_new_tokens"] > model.config.n_positions:
                                raise ValueError("Prompt plus output exceeds model context")
                            with torch.inference_mode():
                                if generation["profile"] in ("lora_paper", "lora_readme"):
                                    from utils.source_beam import generate

                                    result = generate(model, inputs["input_ids"], generation)
                                else:
                                    result = model.generate(
                                        **inputs,
                                        max_new_tokens=generation["max_new_tokens"],
                                        do_sample=False,
                                        num_beams=generation["num_beams"],
                                        length_penalty=generation["length_penalty"],
                                        no_repeat_ngram_size=generation["no_repeat_ngram_size"],
                                        eos_token_id=generation["eos_token_id"],
                                        pad_token_id=generation["pad_token_id"],
                                        bad_words_ids=generation["bad_words_ids"],
                                        repetition_penalty=generation["repetition_penalty"],
                                    )
                            record["prediction"] = clean_prediction(
                                tokenizer.decode(
                                    result[0][length:],
                                    skip_special_tokens=generation["protocol"] == "legacy",
                                ),
                                generation["protocol"],
                            )
                        except Exception as exc:
                            record.update(prediction=None, error=str(exc))
                            generation_errors += 1
                        record["elapsed_seconds"] = time.perf_counter() - sample_started
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        stream.flush()
                        elapsed = time.perf_counter() - started
                        write_json(
                            path / "progress.json",
                            dict(
                                phase="generation",
                                completed=i + 1,
                                total=len(dataset),
                                remaining_seconds=elapsed / (i + 1) * (len(dataset) - i - 1),
                            ),
                        )
                        print(f"Generated {i + 1}/{len(dataset)}", flush=True)
                        if record.get("error") and "out of memory" in record["error"].lower():
                            raise RuntimeError(record["error"])
                if generation_errors:
                    raise RuntimeError(
                        f"{generation_errors} generation errors; see predictions.jsonl"
                    )
            finally:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        generated = time.perf_counter()
        write_json(
            path / "progress.json", dict(phase="scoring", completed=0, total=len(config["metrics"]))
        )
        check_coverage(read_predictions(path / "predictions.jsonl"), dataset)
        write_json(
            path / "coverage.json",
            dict(
                expected=len(dataset),
                actual=len(dataset),
                complete=True,
                full_split=config["samples"] is None,
            ),
        )
        profile = config.get("scoring_profile", "legacy_single")
        if profile == "e2e_official":
            from utils.official_scoring import score_official

            scores = score_official(
                path / "predictions.jsonl", config["metrics"], path, config["local_files_only"]
            )
        elif profile == "legacy_single":
            scores = score(path / "predictions.jsonl", config["metrics"], path)
        else:
            raise ValueError(f"Unknown scoring profile: {profile}")
        write_json(
            path / "timing.json",
            dict(
                generation_or_copy_seconds=generated - started,
                scoring_seconds=time.perf_counter() - generated,
            ),
        )
        write_json(
            path / "progress.json",
            dict(phase="completed", completed=1, total=1, remaining_seconds=0),
        )
        print(scores, flush=True)
    return path
