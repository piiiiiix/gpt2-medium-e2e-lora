"""全参数微调命令行入口：使用独立 FFT 默认配置，调用公共训练后端。"""

from copy import deepcopy

from config import FFT_DEFAULTS

DEFAULTS = deepcopy(FFT_DEFAULTS)

if __name__ == "__main__":
    from utils.cli import arguments
    from utils.training import train

    config, path = arguments(DEFAULTS, "fft")
    train(config, path, "fft")
