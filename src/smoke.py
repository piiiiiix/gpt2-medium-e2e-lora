"""Incremental smoke checks; all artifacts stay under outputs/smoke."""
import argparse
import json
import time
from pathlib import Path
from config import TRAIN_DEFAULTS, require_lora
from utils.runs import new_run, running, read_json

def runtime():
    config={**TRAIN_DEFAULTS,'smoke':True}
    path = new_run('lora', config)
    with running(path, 'lora', config):
        print('runtime smoke passed')
        time.sleep(.1)
    assert read_json(path/'status.json')['status']=='completed'
    assert json.loads((path/'gpu.jsonl').read_text().splitlines()[0])
    failed = new_run('fft', config)
    try:
        with running(failed,'fft',config):
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
        epochs=1,local_files_only=True,smoke=True,ranks=[dict(q=2,k=0,v=1),dict(q=0,k=1,v=0)])
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

def evaluation_pipeline():
    from copy import deepcopy
    from config import OUTPUT_ROOT,EVAL_DEFAULTS,METRICS
    from utils.evaluation import evaluate
    from utils.scoring import score,read_predictions
    folder=OUTPUT_ROOT/'smoke'
    for kind in ['base','lora','fft']:
        path=str(folder/'tiny-base') if kind=='base' else read_json(folder/f'{kind}-run.json')['path']+'/final'
        config=deepcopy(EVAL_DEFAULTS)
        config.update(model=dict(type=kind,path=path),local_data=str(folder/'data.json'),
            samples=2,max_new_tokens=8,num_beams=2,local_files_only=True,metrics=['BLEU','ROUGE-L','CIDEr'],smoke=True)
        run=new_run('benchmarks',config)
        evaluate(config,run)
        assert len(read_predictions(run/'predictions.jsonl'))==2
        original=read_json(run/'metrics.json')['metrics']
        config['predictions']=str(run/'predictions.jsonl')
        repeat=new_run('benchmarks',config)
        evaluate(config,repeat)
        assert read_json(repeat/'metrics.json')['metrics']==original
    fixture=folder/'scores-fixture.jsonl'
    with fixture.open('w',encoding='utf-8') as stream:
        for i in range(4):
            text=f'Restaurant number {i} serves very delicious Italian food near the city centre .'
            stream.write(json.dumps(dict(id=i,target=text,prediction=text,references=[text]))+'\n')
    score(fixture,METRICS,folder)
    assert len(read_json(folder/'metrics.json')['metrics'])==8
    assert all('scores' in r for r in read_predictions(folder/'sample_scores.jsonl'))
    print('PASS: Base/LoRA/FFT generation, incremental rows, offline rescoring, all five metrics and per-sample scores')

def frontend_pipeline():
    import sys
    from config import OUTPUT_ROOT,EVAL_DEFAULTS
    from copy import deepcopy
    from streamlit.testing.v1 import AppTest
    from utils.jobs import launch,tensorboard,alive
    from utils.runs import read_json
    from urllib.request import urlopen
    existing_pids={read_json(p,{}).get('pid') for p in OUTPUT_ROOT.rglob('tensorboard_server.json')}
    app=AppTest.from_file(str(Path(__file__).with_name('app.py')),default_timeout=30).run()
    assert not app.exception
    app.button[1].click().run() # 全部冻结
    assert not app.exception
    assert all(v==0 for row in app.session_state['rank_rows'] for k,v in row.items() if k!='layer')
    for page in ['测试 / benchmark','历史测试记录','任务进度']:
        app.radio(key='page').set_value(page).run()
        assert not app.exception
    config=deepcopy(EVAL_DEFAULTS)
    config.update(model=dict(type='base',path=str(OUTPUT_ROOT/'smoke/tiny-base')),
        local_data=str(OUTPUT_ROOT/'smoke/data.json'),samples=1,max_new_tokens=6,num_beams=1,
        local_files_only=True,metrics=['BLEU'],smoke=True)
    runs=launch([('tests',config),('benchmarks',config)])
    deadline=time.monotonic()+90
    while time.monotonic()<deadline:
        statuses=[read_json(Path(r['path'])/'status.json')['status'] for r in runs]
        if all(s in ('completed','failed') for s in statuses):
            break
        time.sleep(.5)
    assert statuses==['completed','completed'],statuses
    run=read_json(OUTPUT_ROOT/'smoke/lora-run.json')['path']
    url=tensorboard(run)
    deadline=time.monotonic()+30
    success=False
    while time.monotonic()<deadline:
        try:
            with urlopen(url+'/data/plugin/scalars/tags',timeout=2) as response:
                tags=json.loads(response.read())
                assert tags
                success=True
                break
        except OSError:
            time.sleep(.5)
    assert success,'TensorBoard did not expose training scalars'
    # Stop only the helper launched for this smoke check.
    import subprocess
    for server in OUTPUT_ROOT.rglob('tensorboard_server.json'):
        pid=read_json(server,{}).get('pid')
        if pid not in existing_pids and alive(pid):
            subprocess.run(['taskkill','/PID',str(pid),'/T'],capture_output=True)
    print('PASS: four frontend pages, all-frozen control, detached sequential jobs, TensorBoard HTTP scalars')

def medium_pipeline(kind):
    from copy import deepcopy
    from config import OUTPUT_ROOT,TRAIN_DEFAULTS,FFT_DEFAULTS,EVAL_DEFAULTS,MODEL_NAME
    from utils.training import train
    from utils.evaluation import evaluate
    from utils.runs import write_json
    if kind=='base':
        spec=dict(type='base',path=MODEL_NAME,name='GPT-2 Medium real smoke Base')
    else:
        config=deepcopy(TRAIN_DEFAULTS if kind=='lora' else FFT_DEFAULTS)
        config.update(max_steps=1,train_samples=2,eval_samples=2,batch_size=1,eval_batch_size=1,
            gradient_accumulation=1,logging_steps=1,warmup_steps=0,max_length=128,precision='bf16',
            local_files_only=True,smoke=True)
        path=new_run(kind,config)
        train(config,path,kind)
        spec=dict(type=kind,path=str(path/'final'),name=f'GPT-2 Medium real smoke {kind}',training_run=str(path))
        write_json(OUTPUT_ROOT/f'smoke/medium-{kind}-run.json',dict(path=str(path)))
    config=deepcopy(EVAL_DEFAULTS)
    config.update(model=spec,samples=2,max_new_tokens=12,local_files_only=True,smoke=True)
    evaluation=new_run('benchmarks',config)
    evaluate(config,evaluation)
    assert read_json(evaluation/'status.json')['status']=='completed'
    assert len(read_json(evaluation/'metrics.json')['metrics'])==8
    write_json(OUTPUT_ROOT/f'smoke/medium-{kind}-eval.json',dict(path=str(evaluation)))
    print(f'PASS: ACTUAL GPT-2 Medium {kind}, E2E samples, all metrics, telemetry and logs')

def guards_pipeline():
    import torch
    from transformers import GPT2LMHeadModel
    from config import OUTPUT_ROOT,PROJECT_ROOT,inside_project
    from utils.qkv_lora import attach,validate_ranks
    from utils.scoring import score
    from utils.training import Progress
    from utils.runs import estimate
    folder=OUTPUT_ROOT/'smoke/guards'
    folder.mkdir(parents=True,exist_ok=True)
    try:
        inside_project(PROJECT_ROOT.parent/'gpt2-small-e2e-lora')
        raise AssertionError('Outside path accepted')
    except ValueError:
        pass
    model=GPT2LMHeadModel.from_pretrained(OUTPUT_ROOT/'smoke/tiny-base').eval()
    ids=torch.tensor([[1,2,3]])
    expected=model(ids).logits.detach()
    attach(model,[dict(q=0,k=0,v=0) for _ in range(2)])
    assert not any(p.requires_grad for p in model.parameters())
    torch.testing.assert_close(model(ids).logits,expected)
    try:
        validate_ranks([dict(q=-1,k=0,v=0)],1)
        raise AssertionError('Negative rank accepted')
    except ValueError:
        pass
    class State:
        global_step=1
        epoch=1
    try:
        Progress(folder).on_log(None,State(),None,logs={'grad_norm':float('nan')})
        raise AssertionError('NaN gradient accepted')
    except FloatingPointError:
        pass
    prediction=folder/'predictions.jsonl'
    prediction.write_text(json.dumps(dict(id=0,target='food',prediction='',references=['food']))+'\n')
    try:
        score(prediction,['BLEU','NIST'],folder)
        raise AssertionError('Undefined NIST should fail')
    except RuntimeError:
        pass
    result=read_json(folder/'metrics.json')
    assert 'BLEU-4' in result['metrics'] and 'NIST' in result['errors']
    assert prediction.exists() and (folder/'sample_scores.jsonl').exists()
    assert estimate(TRAIN_DEFAULTS,'lora')==(None,0)
    print('PASS: project containment, all-frozen logits, invalid ranks, NaN rejection, partial-score retention, smoke excluded from ETA')

if __name__ == '__main__':
    require_lora()
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['runtime','models','lora','fft','evaluation','frontend',
        'medium-base','medium-lora','medium-fft','guards'])
    args = parser.parse_args()
    {'runtime':runtime,'models':model_pipeline,'lora':lambda:training_pipeline('lora'),
        'fft':lambda:training_pipeline('fft'),'evaluation':evaluation_pipeline,
        'frontend':frontend_pipeline,'medium-base':lambda:medium_pipeline('base'),
        'medium-lora':lambda:medium_pipeline('lora'),'medium-fft':lambda:medium_pipeline('fft'),
        'guards':guards_pipeline}[args.stage]()
