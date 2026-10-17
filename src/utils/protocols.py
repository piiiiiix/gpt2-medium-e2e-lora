"""保存输入协议，按模型解析生成参数；无协议的历史模型保持旧行为。"""

from pathlib import Path

from config import inside_project
from utils.runs import read_json

GENERATION = {
    "legacy": dict(
        protocol="legacy",
        max_new_tokens=64,
        num_beams=10,
        length_penalty=0.9,
        no_repeat_ngram_size=4,
        eos_token_id=None,
    ),
    "lora_paper": dict(
        protocol="lora_e2e",
        max_new_tokens=64,
        num_beams=10,
        length_penalty=0.9,
        no_repeat_ngram_size=4,
        eos_token_id=[50256, 628],
    ),
    "lora_readme": dict(
        protocol="lora_e2e",
        max_new_tokens=64,
        num_beams=10,
        length_penalty=0.8,
        no_repeat_ngram_size=4,
        eos_token_id=[50256, 628],
    ),
    "prefix_ft": dict(
        protocol="prefix_ft",
        max_new_tokens=100,
        num_beams=5,
        length_penalty=1.0,
        no_repeat_ngram_size=0,
        eos_token_id=None,
    ),
}


def model_protocol(spec):
    if spec["type"] == "base":
        return "lora_e2e"
    path = inside_project(spec["path"])
    saved = read_json(path / "training_protocol.json", {})
    if saved:
        return saved["protocol"]
    # Also works for checkpoints and direct CLI model paths, without UI metadata.
    for parent in [path, *path.parents]:
        if (parent / "status.json").exists():
            return read_json(parent / "config.json", {}).get("protocol", "legacy")
    return spec.get("training_config", {}).get("protocol", "legacy")


def resolve_generation(config):
    profile = config.get("generation_profile", "legacy")
    if profile == "auto":
        profile = {"lora_e2e": "lora_paper", "prefix_ft": "prefix_ft", "legacy": "legacy"}[
            model_protocol(config["model"])
        ]
    if profile not in GENERATION:
        raise ValueError(f"Unknown generation profile: {profile}")
    resolved = {**GENERATION[profile], "profile": profile}
    for key in ("max_new_tokens", "num_beams", "length_penalty", "no_repeat_ngram_size"):
        if config.get(key) is not None:
            resolved[key] = config[key]
    if resolved["max_new_tokens"] <= 0 or resolved["num_beams"] <= 0:
        raise ValueError("Generation length and beams must be positive")
    return resolved


def clean_prediction(text, protocol):
    if protocol == "lora_e2e":
        text = text.split("<|endoftext|>")[0].split("\n\n")[0]
    elif protocol == "prefix_ft":
        text = text.split("<|endoftext|>")[0]
    return text.strip()


def save_protocol(path, config):
    from utils.runs import write_json

    write_json(
        Path(path) / "training_protocol.json", {"protocol": config.get("protocol", "legacy")}
    )
