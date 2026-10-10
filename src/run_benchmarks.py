"""Benchmark 命令行入口：选择评测指标，支持生成后评分或使用已有预测重新评分。"""

from copy import deepcopy

from config import EVAL_DEFAULTS

DEFAULTS = deepcopy(EVAL_DEFAULTS)

if __name__ == "__main__":
    from utils.cli import arguments
    from utils.evaluation import evaluate

    config, path = arguments(DEFAULTS, "benchmarks")
    evaluate(config, path, "benchmarks")
