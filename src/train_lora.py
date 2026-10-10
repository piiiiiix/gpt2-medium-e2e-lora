"""LoRA 训练命令行入口：顶部 DEFAULTS 定义默认参数，JSON 配置可覆盖参数。"""

from copy import deepcopy

from config import TRAIN_DEFAULTS

DEFAULTS = deepcopy(TRAIN_DEFAULTS)

if __name__ == "__main__":
    from utils.cli import arguments
    from utils.training import train

    config, path = arguments(DEFAULTS, "lora")
    train(config, path, "lora")
