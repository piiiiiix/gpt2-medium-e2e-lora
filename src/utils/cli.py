"""JSON config overrides for task scripts."""
import argparse
import copy
from config import inside_project
from utils.runs import read_json, new_run

def arguments(defaults, kind):
    parser=argparse.ArgumentParser(description=f'GPT-2 Medium {kind}')
    parser.add_argument('--config',help='JSON overrides, path inside medium project')
    parser.add_argument('--run-dir',help='Existing queued run directory')
    args=parser.parse_args()
    config=copy.deepcopy(defaults)
    if args.config:
        overrides=read_json(inside_project(args.config))
        if not isinstance(overrides,dict):
            raise ValueError('Config must contain a JSON object')
        unknown=set(overrides)-set(config)
        if unknown:
            raise ValueError(f'Unknown config keys: {unknown}')
        config.update(overrides)
    path=inside_project(args.run_dir) if args.run_dir else new_run(kind,config)
    if args.run_dir and read_json(path/'status.json',{}).get('status')!='queued':
        raise ValueError('Only queued run directories may be started; never overwrite a prior run')
    return config,path
