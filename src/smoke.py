"""Incremental smoke checks; all artifacts stay under outputs/smoke."""
import argparse
import json
import time
from config import TRAIN_DEFAULTS, require_lora
from utils.runs import new_run, running, read_json

def runtime():
    path = new_run('lora', TRAIN_DEFAULTS)
    with running(path, 'lora', TRAIN_DEFAULTS):
        print('runtime smoke passed')
        time.sleep(.1)
    assert read_json(path/'status.json')['status']=='completed'
    assert json.loads((path/'gpu.jsonl').read_text().splitlines()[0])
    failed = new_run('fft', TRAIN_DEFAULTS)
    try:
        with running(failed,'fft',TRAIN_DEFAULTS):
            raise RuntimeError('deliberate smoke failure')
    except RuntimeError:
        pass
    assert read_json(failed/'status.json')['status']=='failed'
    print('PASS: runtime success/failure status, log, GPU telemetry')

if __name__ == '__main__':
    require_lora()
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['runtime'])
    args = parser.parse_args()
    runtime()
