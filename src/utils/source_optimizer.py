"""作者旧版 AdamW 的更新顺序：原始二阶矩加 epsilon，最后衰减权重。

与现代 torch AdamW 的 epsilon / weight-decay 顺序不同。公式依据两篇
论文仓库的 optimizer.py / optimization.py；bias correction 始终开启。
"""

import math

import torch


class SourceAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("SourceAdamW requires dense gradients")
                state = self.state[p]
                if not state:
                    state.update(
                        step=0, exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p)
                    )
                state["step"] += 1
                m, v = state["exp_avg"], state["exp_avg_sq"]
                m.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                t = state["step"]
                rate = group["lr"] * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.addcdiv_(m, v.sqrt().add_(group["eps"]), value=-rate)
                p.mul_(1 - group["lr"] * group["weight_decay"])
        return loss
