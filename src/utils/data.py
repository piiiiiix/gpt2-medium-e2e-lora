"""E2E 数据模块：加载数据、选择样本、构造提示词并生成仅目标文本参与损失的标签。"""

import json

from datasets import Dataset, DatasetDict, load_dataset

from config import inside_project


def load_data(config):
    """从项目内 JSON 或 GEM/e2e_nlg 加载训练、验证及测试数据。"""
    if config.get("local_data"):
        path = inside_project(config["local_data"])
        raw = json.loads(path.read_text(encoding="utf-8"))
        return DatasetDict({key: Dataset.from_list(rows) for key, rows in raw.items()})
    return load_dataset(config["dataset"], trust_remote_code=True)


def subset(dataset, count):
    """按正整数样本数截取数据；None 表示全部，空数据会报错。"""
    if count is not None:
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("Sample count must be a positive integer or null")
        dataset = dataset.select(range(min(count, len(dataset))))
    if not len(dataset):
        raise ValueError("Dataset is empty")
    return dataset


def prompt(sample):
    """按 small 项目的约定，将 meaning representation 转为 MR/Text 提示词。"""
    return f"MR: {sample['meaning_representation']}\nText:"


def tokenize(sample, tokenizer, max_length):
    """拼接提示词与目标文本，用 -100 屏蔽提示词损失，并检查上下文长度。"""
    prefix = tokenizer(prompt(sample), add_special_tokens=False)["input_ids"]
    target = tokenizer(" " + sample["target"], add_special_tokens=False)["input_ids"] + [
        tokenizer.eos_token_id
    ]
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
