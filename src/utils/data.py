"""E2E loading and target-only loss; small project's prompt convention."""
import json
from pathlib import Path
from datasets import Dataset, DatasetDict, load_dataset
from config import inside_project

def load_data(config):
    if config.get('local_data'):
        path = inside_project(config['local_data'])
        raw = json.loads(path.read_text(encoding='utf-8'))
        return DatasetDict({key:Dataset.from_list(rows) for key,rows in raw.items()})
    return load_dataset(config['dataset'],trust_remote_code=True)

def subset(dataset, count):
    if count is not None:
        if isinstance(count,bool) or not isinstance(count,int) or count <= 0:
            raise ValueError('Sample count must be a positive integer or null')
        dataset = dataset.select(range(min(count,len(dataset))))
    if not len(dataset):
        raise ValueError('Dataset is empty')
    return dataset

def prompt(sample):
    return f"MR: {sample['meaning_representation']}\nText:"

def tokenize(sample, tokenizer, max_length):
    prefix = tokenizer(prompt(sample),add_special_tokens=False)['input_ids']
    target = tokenizer(' '+sample['target'],add_special_tokens=False)['input_ids'] + [tokenizer.eos_token_id]
    if len(prefix) >= max_length:
        raise ValueError('Prompt consumes max_length; increase max_length instead of silently removing all target labels')
    tokens = (prefix+target)[:max_length]
    return dict(input_ids=tokens, attention_mask=[1]*len(tokens),
        labels=([-100]*len(prefix)+target)[:max_length])
