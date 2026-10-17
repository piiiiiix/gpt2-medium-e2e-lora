"""QKV LoRA 模块：在 GPT-2 合并的 c_attn 上添加独立 Q、K、V 分支。

三个 rank 为模型级配置；rank=0 不挂分支。基础权重冻结，旧 adapter 保持加载兼容。"""

import json
import math
from pathlib import Path

import torch
from torch import nn


def validate_ranks(ranks, layers):
    """校验 QKV rank 并展开模型级配置；逐层列表仅用于兼容旧模型加载。"""
    if isinstance(ranks, dict):
        rows = [ranks] * layers
    else:
        # 列表仅兼容旧模型加载；新训练统一使用一个 QKV rank 字典。
        if not isinstance(ranks, list) or len(ranks) != layers:
            raise ValueError("Expected one QKV rank object")
        rows = ranks
    for row in rows:
        if set(row) != {"q", "k", "v"}:
            raise ValueError("Every rank row must contain q, k, v")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in row.values()):
            raise ValueError("QKV ranks must be nonnegative integers; 0 freezes a projection")
    return rows


class Branch(nn.Module):
    """单个投影的低秩增量分支，由 A、B、dropout 和 alpha/r 缩放组成。"""

    def __init__(self, width, rank, alpha, dropout):
        """创建低秩矩阵，随机初始化 A、将 B 置零，使初始增量为零。"""
        super().__init__()
        self.a = nn.Linear(width, rank, bias=False)
        self.b = nn.Linear(rank, width, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / rank

    def forward(self, x):
        """计算经过 dropout 和低秩矩阵的投影增量，并按 alpha/r 缩放。"""
        return self.b(self.a(self.dropout(x))) * self.scale


class QKVLoRA(nn.Module):
    """包装冻结的合并 QKV 层，只给启用的 Q、K、V 投影添加 LoRA 增量。"""

    def __init__(self, base, ranks, alpha, dropout, shared_dropout=False):
        """根据三个 rank 创建启用分支，并匹配基础层的设备与精度。"""
        super().__init__()
        self.base_layer = base
        self.shared_dropout = nn.Dropout(dropout) if shared_dropout else None
        width = base.weight.shape[0]
        self.branches = nn.ModuleDict(
            {
                key: Branch(width, rank, alpha, 0 if shared_dropout else dropout)
                for key, rank in ranks.items()
                if rank > 0
            }
        )
        self.to(device=base.weight.device, dtype=base.weight.dtype)

    def forward(self, x):
        """将基础输出拆成 QKV，分别叠加启用分支，再拼回原输出形状。"""
        output = self.base_layer(x)
        pieces = output.chunk(3, dim=-1)
        branch_input = self.shared_dropout(x) if self.shared_dropout is not None else x
        return torch.cat(
            [
                piece + self.branches[key](branch_input) if key in self.branches else piece
                for key, piece in zip(("q", "k", "v"), pieces)
            ],
            dim=-1,
        )


def attach(model, ranks, alpha=32, dropout=0.1, shared_dropout=False):
    """冻结基础模型，将 QKV 配置统一应用到各 Transformer 层的注意力投影。"""
    rows = validate_ranks(ranks, len(model.transformer.h))
    if alpha <= 0 or not 0 <= dropout < 1:
        raise ValueError("alpha must be positive; dropout must be in [0,1)")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for layer, row in zip(model.transformer.h, rows):
        layer.attn.c_attn = QKVLoRA(layer.attn.c_attn, row, alpha, dropout, shared_dropout)
    return model


def adapter_state(model):
    """提取 adapter 分支权重并转到 CPU，不包含冻结的基础模型权重。"""
    return {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
        if ".branches." in name
    }


def save_adapter(model, path, config):
    """保存 adapter safetensors 与基础模型、rank、alpha、dropout 配置。"""
    from safetensors.torch import save_file

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    save_file(adapter_state(model), str(path / "adapter_model.safetensors"))
    (path / "adapter_config.json").write_text(
        json.dumps(
            dict(
                format="medium-qkv-lora-v1",
                base_model_name_or_path=config["model"],
                ranks=config["ranks"],
                alpha=config["alpha"],
                dropout=config["dropout"],
                shared_dropout=bool(
                    config.get("source_training") and config.get("protocol") == "lora_e2e"
                ),
            ),
            indent=2,
        ),
        encoding="utf-8",
    )


def restore(model, path):
    """校验 adapter 权重键与当前分支一致，再恢复分支参数。"""
    from safetensors.torch import load_file

    weights = load_file(str(Path(path) / "adapter_model.safetensors"))
    expected = set(adapter_state(model))
    if set(weights) != expected:
        raise ValueError("Adapter weights do not match the configured QKV ranks")
    model.load_state_dict(weights, strict=False)


def load_adapter(model, path):
    """读取 adapter 配置、重建分支并加载权重，兼容旧逐层配置。"""
    config = json.loads((Path(path) / "adapter_config.json").read_text())
    attach(
        model,
        config["ranks"],
        config["alpha"],
        config["dropout"],
        config.get("shared_dropout", False),
    )
    restore(model, path)
    return model
