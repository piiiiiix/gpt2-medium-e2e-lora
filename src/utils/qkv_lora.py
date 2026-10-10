"""Independent per-layer Q/K/V LoRA on GPT-2's fused Conv1D.

Adapter format is project-specific, not a PEFT adapter. Base weights never change.
"""
import json
import math
from pathlib import Path
import torch
from torch import nn

def validate_ranks(ranks, layers):
    if len(ranks) != layers:
        raise ValueError(f'Expected {layers} QKV rank rows, got {len(ranks)}')
    for row in ranks:
        if set(row) != {'q','k','v'}:
            raise ValueError('Every rank row must contain q, k, v')
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in row.values()):
            raise ValueError('QKV ranks must be nonnegative integers; 0 freezes a projection')

class Branch(nn.Module):
    def __init__(self, width, rank, alpha, dropout):
        super().__init__()
        self.a = nn.Linear(width, rank, bias=False)
        self.b = nn.Linear(rank, width, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / rank

    def forward(self, x):
        return self.b(self.a(self.dropout(x))) * self.scale

class QKVLoRA(nn.Module):
    def __init__(self, base, ranks, alpha, dropout):
        super().__init__()
        self.base_layer = base
        width = base.weight.shape[0]
        self.branches = nn.ModuleDict({key:Branch(width, rank, alpha, dropout)
            for key,rank in ranks.items() if rank > 0})
        self.to(device=base.weight.device, dtype=base.weight.dtype)

    def forward(self, x):
        output = self.base_layer(x)
        pieces = output.chunk(3, dim=-1)
        return torch.cat([piece + self.branches[key](x) if key in self.branches else piece
            for key,piece in zip(('q','k','v'),pieces)], dim=-1)

def attach(model, ranks, alpha=32, dropout=.1):
    validate_ranks(ranks, len(model.transformer.h))
    if alpha <= 0 or not 0 <= dropout < 1:
        raise ValueError('alpha must be positive; dropout must be in [0,1)')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for layer, row in zip(model.transformer.h,ranks):
        layer.attn.c_attn = QKVLoRA(layer.attn.c_attn,row,alpha,dropout)
    return model

def adapter_state(model):
    return {name:tensor.detach().cpu().contiguous()
        for name,tensor in model.state_dict().items() if '.branches.' in name}

def save_adapter(model, path, config):
    from safetensors.torch import save_file
    path = Path(path)
    path.mkdir(parents=True,exist_ok=True)
    save_file(adapter_state(model), str(path/'adapter_model.safetensors'))
    (path/'adapter_config.json').write_text(json.dumps(dict(
        format='medium-qkv-lora-v1', base_model_name_or_path=config['model'],
        ranks=config['ranks'], alpha=config['alpha'], dropout=config['dropout']),indent=2),encoding='utf-8')

def restore(model, path):
    from safetensors.torch import load_file
    weights = load_file(str(Path(path)/'adapter_model.safetensors'))
    expected = set(adapter_state(model))
    if set(weights) != expected:
        raise ValueError('Adapter weights do not match the configured QKV ranks')
    model.load_state_dict(weights,strict=False)

def load_adapter(model, path):
    config = json.loads((Path(path)/'adapter_config.json').read_text())
    attach(model,config['ranks'],config['alpha'],config['dropout'])
    restore(model,path)
    return model
