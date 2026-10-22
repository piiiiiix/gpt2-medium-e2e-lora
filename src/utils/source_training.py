"""LoRA source loss, selectable padding and full-batch tail sampling."""

import random

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


class SourceSampler(DistributedSampler):
    """LoRA training epochs are numbered from 1; validation sampler remains at 0."""

    def set_epoch(self, epoch):
        super().set_epoch(epoch + 1)


def source_loss(logits, labels, smoothing):
    logits, labels = logits[:, :-1, :].contiguous(), labels[:, 1:].contiguous()
    mask = labels.ne(-100)
    labels = labels.masked_fill(~mask, 0)
    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
    nll = -logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    loss = (1 - smoothing) * nll - smoothing * logprobs.mean(-1)
    return (loss * mask).sum() / (mask.sum() + 0.0001)


class FullBatchDataset(Dataset):
    """FT_Dataset fills final batch via Random(911), without deleting real examples."""

    def __init__(self, dataset, batch):
        self.dataset, self.batch = dataset, batch
        self.rng = random.Random(911)

    def __len__(self):
        return ((len(self.dataset) + self.batch - 1) // self.batch) * self.batch

    def __getitem__(self, index):
        if index >= len(self.dataset):
            index = self.rng.randint(0, len(self.dataset) - 1)
        return self.dataset[index]


class SourceCollator:
    def __init__(self, length, padding_mode="fixed"):
        self.length = length
        if padding_mode not in ("fixed", "dynamic"):
            raise ValueError("padding_mode must be fixed or dynamic")
        self.padding_mode = padding_mode

    def __call__(self, rows):
        # Official padding id=0; causal attention makes padded future tokens irrelevant.
        longest = max(len(row["input_ids"]) for row in rows)
        if longest > self.length:
            raise ValueError("Tokenized input exceeds max_length")
        length = self.length if self.padding_mode == "fixed" else longest
        return {
            key: torch.tensor([r[key] + [pad] * (length - len(r[key])) for r in rows])
            for key, pad in (("input_ids", 0), ("labels", -100))
        }
