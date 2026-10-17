"""E2E 数据模块：加载数据、选择样本、构造提示词并生成仅目标文本参与损失的标签。"""

import json
import re

from datasets import Dataset, DatasetDict, load_dataset

from config import inside_project


def load_data(config):
    """从项目内 JSON 或 GEM/e2e_nlg 加载训练、验证及测试数据。"""
    if config.get("local_data"):
        path = inside_project(config["local_data"])
        raw = json.loads(path.read_text(encoding="utf-8"))
        return DatasetDict({key: Dataset.from_list(rows) for key, rows in raw.items()})
    if config["dataset"] == "lora/e2e-original":
        return original_data(config.get("local_files_only", False))
    return load_dataset(config["dataset"], trust_remote_code=True)


def original_data(offline=False):
    """Keep official train/valid reference rows; expose grouped splits for generation."""
    from utils.paper_resources import data_files

    result = {}
    for split, path in data_files(offline).items():
        rows, groups = [], {}
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            # Identical to format_converting_e2e.py: context spacing is meaningful for BPE.
            ctx, target = line.strip().split("||")
            target = target.strip("\n")
            rows.append(
                dict(
                    id=f"original-{split}-row-{i}",
                    context=ctx,
                    meaning_representation=ctx,
                    target=target,
                    references=[target],
                )
            )
            clean = target.split("<|endoftext|>")[0].split("\n\n")[0].strip()
            groups.setdefault(ctx, []).append(clean)
        grouped = [
            dict(
                id=f"original-{split}-{i}",
                context=ctx,
                meaning_representation=ctx,
                target=refs[0],
                references=refs,
            )
            for i, (ctx, refs) in enumerate(groups.items())
        ]
        result[split] = Dataset.from_list(grouped if split == "test" else rows)
        if split == "validation":
            result["validation_generation"] = Dataset.from_list(grouped)
    return DatasetDict(result)


def evaluation_data(config):
    raw = load_data(config)
    split = config["split"]
    if split == "validation" and "validation_generation" in raw:
        split = "validation_generation"
    return subset(raw[split], config["samples"])


def subset(dataset, count):
    """按正整数样本数截取数据；None 表示全部，空数据会报错。"""
    if count is not None:
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("Sample count must be a positive integer or null")
        dataset = dataset.select(range(min(count, len(dataset))))
    if not len(dataset):
        raise ValueError("Dataset is empty")
    return dataset


def context(sample):
    """保留原始 context；GEM MR 仅转换字段名，不冒充原始数据集。"""
    if "context" in sample:
        return sample["context"]
    mr = sample["meaning_representation"]
    fields = re.findall(r"([^,\[\]]+)\[([^\]]*)\]", mr)
    if not fields or re.sub(r"[^,\[\]]+\[[^\]]*\]", "", mr).strip(" ,"):
        raise ValueError("Expected E2E slot[value] MR or an explicit context field")
    names = {
        "eatType": "Type",
        "priceRange": "price",
        "customer rating": "customer rating",
        "customerRating": "customer rating",
        "familyFriendly": "family friendly",
    }
    return " | ".join(f"{names.get(k.strip(), k.strip())} : {v}" for k, v in fields)


def prompt(sample, protocol="legacy", tokenizer=None):
    """按模型训练协议构造提示词；legacy 保留旧 MR/Text 格式。"""
    if protocol == "legacy":
        return f"MR: {sample['meaning_representation']}\nText:"
    if tokenizer is None:
        raise ValueError("Source protocols require a tokenizer")
    if protocol == "lora_e2e":
        return context(sample) + tokenizer.bos_token
    if protocol == "prefix_ft":
        return f" {context(sample)} {tokenizer.bos_token}"
    raise ValueError(f"Unknown protocol: {protocol}")


def tokenize(sample, tokenizer, max_length, protocol="legacy"):
    """拼接提示词与目标文本，用 -100 屏蔽提示词损失，并检查上下文长度。"""
    prefix = tokenizer(prompt(sample, protocol, tokenizer), add_special_tokens=False)["input_ids"]
    text = " " + sample["target"]
    if protocol == "prefix_ft":
        text += " "
    target = tokenizer(text, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    if len(prefix) >= max_length:
        raise ValueError(
            "Prompt consumes max_length; increase max_length "
            "instead of silently removing all target labels"
        )
    tokens = (prefix + target)[:max_length]
    return dict(
        input_ids=tokens,
        attention_mask=[1] * len(tokens),
        labels=([-100] * len(prefix) + target)[:max_length],
    )
