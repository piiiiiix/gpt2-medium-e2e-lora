"""Standalone LoRA training. Edit defaults here or pass JSON overrides."""
from copy import deepcopy
from config import TRAIN_DEFAULTS
DEFAULTS=deepcopy(TRAIN_DEFAULTS)

if __name__=='__main__':
    from utils.cli import arguments
    from utils.training import train
    config,path=arguments(DEFAULTS,'lora')
    train(config,path,'lora')
