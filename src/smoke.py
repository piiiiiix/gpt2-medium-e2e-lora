"""Incremental smoke checks; all artifacts stay under outputs/smoke."""
import argparse
import json
import time
from pathlib import Path
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

def model_pipeline():
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
    from utils.qkv_lora import attach, save_adapter, load_adapter
    from utils.data import tokenize
    from utils.models import discover
    from utils.runs import write_json
    from config import OUTPUT_ROOT
    folder=OUTPUT_ROOT/'smoke'
    folder.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(42)
    tokenizer=AutoTokenizer.from_pretrained('openai-community/gpt2',local_files_only=True)
    tokenizer.pad_token=tokenizer.eos_token
    tiny=GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer),n_layer=2,n_head=2,n_embd=32,n_positions=128,
        eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.eos_token_id))
    tiny.save_pretrained(folder/'tiny-base')
    tokenizer.save_pretrained(folder/'tiny-base')
    rows=[dict(meaning_representation=f'name[Restaurant {i}], food[Italian]',
        target=f'Restaurant {i} serves Italian food.',references=[f'Restaurant {i} serves Italian food.']) for i in range(4)]
    write_json(folder/'data.json',dict(train=rows,validation=rows,test=rows))
    assert any(x!=-100 for x in tokenize(rows[0],tokenizer,128)['labels'])
    ranks=[dict(q=2,k=0,v=1),dict(q=0,k=3,v=0)]
    model=attach(tiny,ranks,alpha=32,dropout=0)
    assert 'k' not in model.transformer.h[0].attn.c_attn.branches
    assert all(not p.requires_grad for n,p in model.named_parameters() if '.branches.' not in n)
    ids=torch.tensor([[1,2,3,4]])
    model(ids,labels=ids).loss.backward()
    assert all(p.grad is None for n,p in model.named_parameters() if '.branches.' not in n)
    with torch.no_grad():
        for n,p in model.named_parameters():
            if '.branches.' in n:
                p.add_(.01)
    model.eval()
    expected=model(ids).logits.detach()
    save_adapter(model,folder/'adapter',dict(model=str(folder/'tiny-base'),ranks=ranks,alpha=32,dropout=0))
    reloaded=load_adapter(GPT2LMHeadModel.from_pretrained(folder/'tiny-base'),folder/'adapter').eval()
    torch.testing.assert_close(reloaded(ids).logits,expected)
    assert any(m['path']==str(folder/'adapter') for m in discover())
    print('PASS: tokenization, per-layer QKV freezing, backward, adapter save/reload, discovery')

def training_pipeline(kind):
    from copy import deepcopy
    from config import OUTPUT_ROOT
    from utils.training import train
    from utils.models import load_model
    from utils.runs import write_json
    config=deepcopy(TRAIN_DEFAULTS)
    config.update(model=str(OUTPUT_ROOT/'smoke/tiny-base'),local_data=str(OUTPUT_ROOT/'smoke/data.json'),
        max_steps=2,batch_size=2,eval_batch_size=2,logging_steps=1,warmup_steps=0,max_length=128,
        epochs=1,local_files_only=True,ranks=[dict(q=2,k=0,v=1),dict(q=0,k=1,v=0)])
    path=new_run(kind,config)
    train(config,path,kind)
    assert read_json(path/'status.json')['status']=='completed'
    assert list((path/'tensorboard').glob('events.*'))
    assert read_json(path/'result.json')['best_checkpoint']
    model,tokenizer=load_model(dict(type=kind,path=str(path/'final')),True,device='cpu')
    output=model.generate(**tokenizer('MR: name[Test]\nText:',return_tensors='pt'),max_new_tokens=2,pad_token_id=tokenizer.eos_token_id)
    assert output.shape[1]>0
    write_json(OUTPUT_ROOT/f'smoke/{kind}-run.json',{'path':str(path)})
    print(f'PASS: {kind} training, evaluation, checkpoint, best reload, TensorBoard, final generation')

if __name__ == '__main__':
    require_lora()
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['runtime','models','lora','fft'])
    args = parser.parse_args()
    {'runtime':runtime,'models':model_pipeline,'lora':lambda:training_pipeline('lora'),
        'fft':lambda:training_pipeline('fft')}[args.stage]()
