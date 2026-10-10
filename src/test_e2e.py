"""E2E 测试命令行入口：调用公共生成与评分流程，输出保存在 tests 目录。"""

from copy import deepcopy

from config import EVAL_DEFAULTS

DEFAULTS = deepcopy(EVAL_DEFAULTS)

if __name__ == "__main__":
    from utils.cli import arguments
    from utils.evaluation import evaluate

    config, path = arguments(DEFAULTS, "tests")
    evaluate(config, path, "tests")
