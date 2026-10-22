"""Four-path padding smoke and a short real-Medium LoRA timing probe; no full-score claim."""

import argparse
import statistics
from copy import deepcopy

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from config import FFT_DEFAULTS, MODEL_NAME, OUTPUT_ROOT, TRAIN_DEFAULTS
from utils.models import load_model
from utils.runs import new_run, read_json, write_json
from utils.training import train


def run(kind, mode, model, data, length, steps):
    config = deepcopy(TRAIN_DEFAULTS if kind == "lora" else FFT_DEFAULTS)
    config.update(
        model=model,
        local_data=data,
        padding_mode=mode,
        max_length=length,
        max_steps=steps,
        train_samples=8,
        eval_samples=2,
        batch_size=1,
        eval_batch_size=1,
        logging_steps=1,
        warmup_steps=0,
        local_files_only=True,
        smoke=True,
    )
    path = new_run(kind, config)
    train(config, path, kind)
    result = read_json(path / "result.json")
    assert read_json(path / "status.json")["status"] == "completed"
    assert result["weight_update_verified"]
    assert read_json(path / "data_summary.json")["padding_mode"] == mode
    assert read_json(path / "config.json")["padding_mode"] == mode
    events = EventAccumulator(str(path / "tensorboard"), size_guidance={"scalars": 0})
    events.Reload()
    loss = events.Scalars("train/loss")
    durations = [(b.wall_time - a.wall_time) / (b.step - a.step) for a, b in zip(loss, loss[1:])]
    evidence = dict(
        path=str(path),
        config=config,
        result=result,
        logged_step_seconds=durations,
        median_logged_step_seconds=statistics.median(durations),
    )
    write_json(OUTPUT_ROOT / f"smoke/padding-{kind}-{mode}-{length}.json", evidence)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["tiny", "medium-fixed", "medium-dynamic"])
    stage = parser.parse_args().stage
    torch.set_num_threads(2)
    folder = OUTPUT_ROOT / "smoke/padding-fixture"
    folder.mkdir(parents=True, exist_ok=True)
    if stage == "tiny":
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
        tokenizer.pad_token = tokenizer.eos_token
        tiny = GPT2LMHeadModel(
            GPT2Config(vocab_size=len(tokenizer), n_layer=2, n_head=2, n_embd=32, n_positions=128)
        )
        tiny.save_pretrained(folder / "base")
        tokenizer.save_pretrained(folder / "base")
        rows = [
            dict(context="name : Test | Type : pub", target="Test is a pub."),
            dict(context="name : Blue | food : Italian", target="Blue serves Italian food."),
        ]
        write_json(folder / "data.json", dict(train=rows * 4, validation=rows, test=rows))
        for kind in ("lora", "fft"):
            for mode in ("fixed", "dynamic"):
                path = run(kind, mode, str(folder / "base"), str(folder / "data.json"), 128, 2)
                model, tok = load_model(
                    dict(type=kind, path=str(path / "final")), True, device="cpu"
                )
                output = model.generate(**tok("Test", return_tensors="pt"), max_new_tokens=2)
                assert output.shape[1] > 1
                del model
        print("PASS: LoRA/FT x fixed/dynamic: weight update, config, final reload, generation")
    else:
        mode = stage.split("-")[1]
        path = run("lora", mode, MODEL_NAME, None, 512, 4)
        print(f"PASS: real Medium LoRA {mode}, FP32 batch1, 4 steps: {path}")
