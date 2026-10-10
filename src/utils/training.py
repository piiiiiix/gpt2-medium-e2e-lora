"""Shared Trainer pipeline, with project-specific QKV adapter checkpoints."""
import json
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments, TrainerCallback, set_seed
from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
from utils.data import load_data, subset, tokenize
from utils.qkv_lora import attach, save_adapter, restore
from utils.runs import running, write_json

class Progress(TrainerCallback):
    def __init__(self,path):
        self.path=path
        self.started=time.perf_counter()
    def on_log(self,args,state,control,logs=None,**kwargs):
        with (self.path/'training.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps({'step':state.global_step,'epoch':state.epoch,**(logs or {})},default=str)+'\n')
    def on_step_end(self,args,state,control,**kwargs):
        elapsed=time.perf_counter()-self.started
        write_json(self.path/'progress.json',dict(completed=state.global_step,total=state.max_steps,
            phase='training',remaining_seconds=elapsed/state.global_step*(state.max_steps-state.global_step)))

class AdapterTrainer(Trainer):
    def __init__(self,*args,run_config=None,**kwargs):
        self.run_config=run_config
        super().__init__(*args,**kwargs)
    def _save(self,output_dir=None,state_dict=None):
        save_adapter(self.model,output_dir or self.args.output_dir,self.run_config)
        if self.processing_class:
            self.processing_class.save_pretrained(output_dir or self.args.output_dir)
    def _load_best_model(self):
        if self.state.best_model_checkpoint:
            restore(self.model,self.state.best_model_checkpoint)

def validate(config,kind):
    for key in ['batch_size','eval_batch_size','gradient_accumulation','logging_steps','max_length']:
        if not isinstance(config[key],int) or config[key]<=0:
            raise ValueError(f'{key} must be a positive integer')
    if config['epochs']<=0 or config['learning_rate']<=0:
        raise ValueError('epochs and learning rate must be positive')
    if config['precision'] not in ('fp32','fp16','bf16'):
        raise ValueError('precision must be fp32/fp16/bf16')
    if config['precision']!='fp32' and not torch.cuda.is_available():
        raise ValueError('Select fp32 on CPU')
    if not 0<=config['label_smoothing']<1:
        raise ValueError('label_smoothing must be in [0,1)')
    if kind=='lora' and not any(v>0 for row in config['ranks'] for v in row.values()):
        raise ValueError('All QKV projections are frozen. Use Base testing, or enable at least one rank for training.')

def train(config,path,kind):
    with running(path,kind,config):
        validate(config,kind)
        set_seed(config['seed'])
        tokenizer=AutoTokenizer.from_pretrained(config['model'],local_files_only=config['local_files_only'])
        tokenizer.pad_token=tokenizer.eos_token
        raw=load_data(config)
        train_data=subset(raw['train'],config['train_samples'])
        eval_data=subset(raw['validation'],config['eval_samples'])
        write_json(path/'data_summary.json',dict(dataset=config['dataset'],local_data=config['local_data'],
            train_count=len(train_data),validation_count=len(eval_data),train_fingerprint=train_data._fingerprint,
            validation_fingerprint=eval_data._fingerprint,prompt='MR: {meaning_representation}\nText:',
            loss='target only',examples=[train_data[i] for i in range(min(3,len(train_data)))]))
        writer=SummaryWriter(log_dir=str(path/'tensorboard'))
        try:
            writer.add_text('configuration',json.dumps(config,indent=2),0)
            writer.add_text('training_data', (path/'data_summary.json').read_text(),0)
            encode=lambda row:tokenize(row,tokenizer,config['max_length'])
            train_data=train_data.map(encode,remove_columns=train_data.column_names)
            eval_data=eval_data.map(encode,remove_columns=eval_data.column_names)
            model=AutoModelForCausalLM.from_pretrained(config['model'],local_files_only=config['local_files_only'],
                resid_pdrop=config['dropout'],embd_pdrop=config['dropout'],attn_pdrop=config['dropout'])
            if config['max_length']>model.config.n_positions:
                raise ValueError('max_length exceeds model context window')
            if kind=='lora':
                attach(model,config['ranks'],config['alpha'],config['dropout'])
                if config['gradient_checkpointing']:
                    model.enable_input_require_grads()
            write_json(path/'parameters.json',dict(total=sum(p.numel() for p in model.parameters()),
                trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)))
            args=TrainingArguments(output_dir=str(path/'checkpoints'),
                per_device_train_batch_size=config['batch_size'],per_device_eval_batch_size=config['eval_batch_size'],
                gradient_accumulation_steps=config['gradient_accumulation'],learning_rate=config['learning_rate'],
                num_train_epochs=config['epochs'],max_steps=config['max_steps'],optim='adamw_torch',
                lr_scheduler_type='linear',weight_decay=config['weight_decay'],warmup_steps=config['warmup_steps'],
                label_smoothing_factor=config['label_smoothing'],logging_steps=config['logging_steps'],
                eval_strategy='epoch',save_strategy='epoch',save_total_limit=None,load_best_model_at_end=True,
                metric_for_best_model='eval_loss',greater_is_better=False,report_to='none',
                fp16=config['precision']=='fp16',bf16=config['precision']=='bf16',
                gradient_checkpointing=config['gradient_checkpointing'],
                gradient_checkpointing_kwargs={'use_reentrant':False},seed=config['seed'],data_seed=config['seed'],
                disable_tqdm=True)
            trainer_class=AdapterTrainer if kind=='lora' else Trainer
            extras={'run_config':config} if kind=='lora' else {}
            trainer=trainer_class(model=model,args=args,processing_class=tokenizer,
                train_dataset=train_data,eval_dataset=eval_data,
                data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer,padding=True,label_pad_token_id=-100),
                callbacks=[TensorBoardCallback(tb_writer=writer),Progress(path)],**extras)
            try:
                result=trainer.train()
                trainer.save_model(str(path/'final'))
                tokenizer.save_pretrained(path/'final')
                write_json(path/'result.json',dict(metrics=result.metrics,best_checkpoint=trainer.state.best_model_checkpoint,
                    best_eval_loss=trainer.state.best_metric,final=str(path/'final')))
            finally:
                write_json(path/'train_log.json',trainer.state.log_history)
                writer.flush()
        finally:
            writer.close()
        del trainer,model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return path
