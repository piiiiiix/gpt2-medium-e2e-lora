"""Project-scoped model discovery and common model loading."""
import os
from pathlib import Path
from config import PROJECT_ROOT, MODEL_NAME, inside_project
from utils.runs import read_json

def discover():
    models = [dict(type='base',path=MODEL_NAME,name='GPT-2 Medium (Base)')]
    for directory, folders, files in os.walk(PROJECT_ROOT,followlinks=False):
        folders[:] = [f for f in folders if f not in ('.git','nltk_data','notebooks','__pycache__')
            and (Path(directory)/f).resolve().is_relative_to(PROJECT_ROOT)]
        path = Path(directory)
        if 'adapter_config.json' in files and any(f.startswith('adapter_model.') for f in files):
            kind='lora'
        elif 'config.json' in files and any(f.endswith(('.safetensors','.bin','.index.json')) and 'model' in f for f in files):
            kind='fft'
        else:
            continue
        training = next((p for p in [path,*path.parents] if (p/'status.json').exists()),None)
        metadata = read_json(training/'status.json',{}) if training else {}
        models.append(dict(type=kind,path=str(path),name=str(path.relative_to(PROJECT_ROOT)),
            training_run=str(training) if training else None, trained=metadata.get('started'),
            training_config=read_json(training/'config.json',{}) if training else {}))
    return models

def load_model(spec, local_files_only=False, device=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kind=spec['type']
    if kind not in ('base','lora','fft'):
        raise ValueError('Unknown model type')
    path=spec['path'] if kind=='base' else str(inside_project(spec['path']))
    options=dict(local_files_only=local_files_only)
    if kind=='lora':
        config=read_json(Path(path)/'adapter_config.json')
        if not config:
            raise ValueError('Missing adapter configuration')
        base=config['base_model_name_or_path']
        tokenizer=AutoTokenizer.from_pretrained(path if (Path(path)/'tokenizer_config.json').exists() else base,**options)
        model=AutoModelForCausalLM.from_pretrained(base,**options)
        if config.get('format')=='medium-qkv-lora-v1':
            from utils.qkv_lora import load_adapter
            model=load_adapter(model,path)
        else:
            from peft import PeftModel
            model=PeftModel.from_pretrained(model,path)
    else:
        tokenizer=AutoTokenizer.from_pretrained(path,**options)
        model=AutoModelForCausalLM.from_pretrained(path,**options)
    tokenizer.pad_token=tokenizer.eos_token
    return model.to(device or ('cuda' if torch.cuda.is_available() else 'cpu')), tokenizer
