"""Standalone E2E test entry point; shares generation and scoring profile."""
from copy import deepcopy
from config import EVAL_DEFAULTS
DEFAULTS=deepcopy(EVAL_DEFAULTS)

if __name__=='__main__':
    from utils.cli import arguments
    from utils.evaluation import evaluate
    config,path=arguments(DEFAULTS,'tests')
    evaluate(config,path,'tests')
