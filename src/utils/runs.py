"""Durable run metadata, logging, history and configuration-aware estimates."""
import hashlib
import json
import os
import sys
import time
import traceback
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime, timezone, timedelta
from pathlib import Path
from config import OUTPUT_ROOT, inside_project, require_lora
from utils.gpu_monitor import GPUMonitor

HK = timezone(timedelta(hours=8))

def now():
    return datetime.now(HK).isoformat()

def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    os.replace(temporary, path)

def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return default

def digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:8]

def new_run(kind, config):
    import re
    training = kind in ('lora', 'fft')
    folder = {'lora':'gpt2-e2e-lora', 'fft':'gpt2-e2e-full-ft'}.get(kind, kind)
    if training:
        tag = f"lr{config['learning_rate']:g}_e{config['epochs']}_b{config['batch_size']}_ga{config['gradient_accumulation']}_s{config['seed']}"
        if kind == 'lora':
            tag += '_qkv' + digest(config['ranks'])
    else:
        tag = f"{config['model']['type']}_{digest(config['model'])}_{config['split']}_n{config['samples'] or 'all'}_t{config['max_new_tokens']}_beam{config['num_beams']}_" + '-'.join(config['metrics'])
    tag = re.sub(r'[^a-zA-Z0-9_.-]', '-', tag)
    path = OUTPUT_ROOT / folder / (datetime.now(HK).strftime('%Y%m%d_%H%M%S_%f') + '_' + tag + '_' + digest(config))
    path.mkdir(parents=True, exist_ok=False)
    write_json(path / 'config.json', config)
    write_json(path / 'status.json', dict(kind=kind, status='queued', created=now()))
    return path

class Tee:
    def __init__(self, console, file):
        self.console, self.file = console, file
    def write(self, text):
        self.console.write(text)
        self.file.write(text)
        self.file.flush()
    def flush(self):
        self.console.flush()
        self.file.flush()
    def isatty(self):
        return False

@contextmanager
def running(path, kind, config):
    require_lora()
    path = inside_project(path)
    metadata = dict(kind=kind, status='running', started=now(), pid=os.getpid(),
        config_hash=digest(config), hardware=GPUMonitor.sample(), python=sys.executable)
    write_json(path / 'status.json', metadata)
    import importlib.metadata
    versions = {}
    for package in ['torch', 'transformers', 'datasets', 'peft', 'streamlit', 'tensorboard', 'nltk', 'pycocoevalcap']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    write_json(path / 'environment.json', versions)
    monitor = GPUMonitor(path / 'gpu.jsonl')
    started = time.perf_counter()
    with (path / 'run.log').open('a', encoding='utf-8', buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            monitor.start()
            try:
                print(json.dumps(config, ensure_ascii=False, default=str), flush=True)
                yield
                metadata['status'] = 'completed'
            except BaseException as exc:
                metadata.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
                traceback.print_exc()
                raise
            finally:
                monitor.stop()
                metadata.update(ended=now(), elapsed_seconds=time.perf_counter()-started)
                write_json(path / 'status.json', metadata)

def history():
    records = []
    if OUTPUT_ROOT.exists():
        for path in OUTPUT_ROOT.rglob('status.json'):
            try:
                inside_project(path)
            except ValueError:
                continue
            status = read_json(path)
            if status:
                records.append(dict(path=str(path.parent), **status, config=read_json(path.parent/'config.json', {})))
    return sorted(records, key=lambda x:x['path'], reverse=True)

def estimate(config, kind):
    keys = ['model','dataset','local_data','epochs','batch_size','gradient_accumulation',
        'max_steps','train_samples','eval_samples','precision','gradient_checkpointing','max_length',
        'ranks','split','samples','max_new_tokens','num_beams','metrics','predictions']
    signature = {k:config.get(k) for k in keys}
    current_gpu = [r.get('uuid') for r in GPUMonitor.sample()]
    durations = [r['elapsed_seconds'] for r in history()
        if r['status']=='completed' and r['kind']==kind
        and {k:r['config'].get(k) for k in keys}==signature
        and [g.get('uuid') for g in r.get('hardware',[])]==current_gpu]
    return (sum(durations)/len(durations), len(durations)) if durations else (None, 0)
