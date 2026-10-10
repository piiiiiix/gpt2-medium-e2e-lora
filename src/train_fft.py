"""Standalone full fine-tuning. Defaults are the separate FFT comparison profile."""
from copy import deepcopy
from config import FFT_DEFAULTS
DEFAULTS=deepcopy(FFT_DEFAULTS)

if __name__=='__main__':
    from utils.cli import arguments
    from utils.training import train
    config,path=arguments(DEFAULTS,'fft')
    train(config,path,'fft')
