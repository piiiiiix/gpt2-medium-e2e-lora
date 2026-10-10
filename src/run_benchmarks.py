"""Standalone selected E2E benchmark metrics / offline rescoring."""
from copy import deepcopy
from config import EVAL_DEFAULTS
DEFAULTS=deepcopy(EVAL_DEFAULTS)

if __name__=='__main__':
    from utils.cli import arguments
    from utils.evaluation import evaluate
    config,path=arguments(DEFAULTS,'benchmarks')
    evaluate(config,path,'benchmarks')
