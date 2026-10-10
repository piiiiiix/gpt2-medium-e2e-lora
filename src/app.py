"""Single-file Streamlit frontend; training and tests stay in detached backends."""
import json
import sys
from pathlib import Path
from copy import deepcopy
from datetime import datetime,timedelta
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pandas as pd
import streamlit as st
from config import PROJECT_ROOT,TRAIN_DEFAULTS,FFT_DEFAULTS,EVAL_DEFAULTS,METRICS,require_lora
from utils.models import discover
from utils.runs import history,read_json,estimate,HK
from utils.jobs import launch,tensorboard,alive

require_lora()
st.set_page_config(page_title='GPT-2 Medium · E2E',layout='wide')
st.title('GPT-2 Medium · E2E')
st.caption('训练、评测与实验记录 · conda lora')
page=st.sidebar.radio('菜单',['训练','测试 / benchmark','历史测试记录','任务进度'],key='page')
st.sidebar.caption(str(PROJECT_ROOT))

@st.cache_data(ttl=5,max_entries=2)
def models():
    return discover()

@st.cache_data(ttl=2,max_entries=2)
def records():
    return history()

def start(tasks):
    try:
        runs=launch(tasks)
        st.session_state['last_runs']=runs
        records.clear()
        st.success(f'已启动 {len(runs)} 个任务，将按顺序执行。请到“任务进度”查看。')
    except Exception as exc:
        st.error(str(exc))

def tb_button(model,key):
    run=model.get('training_run')
    if run:
        st.caption(f"训练运行：{Path(run).name}")
        if st.button('打开训练 TensorBoard',key=key):
            try:
                st.session_state[key+'_url']=tensorboard(run)
            except Exception as exc:
                st.error(str(exc))
        if key+'_url' in st.session_state:
            st.link_button('进入 TensorBoard',st.session_state[key+'_url'])
        with st.expander('训练数据与配置'):
            st.json(read_json(Path(run)/'data_summary.json',{}))
            st.json(model.get('training_config',{}))
    elif model['type']!='base':
        st.caption('未找到关联训练运行或 TensorBoard。')

def duration(config,kind):
    if st.button('按历史均值估算耗时',key='estimate_'+kind):
        seconds,count=estimate(config,kind)
        if seconds is None:
            st.info('暂无同配置、同硬件历史估算。')
        else:
            eta=datetime.now(HK)+timedelta(seconds=seconds)
            st.info(f'历史均值 {seconds/60:.1f} 分钟（{count} 次）；预计完成 {eta:%Y-%m-%d %H:%M:%S}。')

if page=='训练':
    kind=st.segmented_control('训练方式',['lora','fft'],default='lora',key='train_kind') or 'lora'
    config=deepcopy(TRAIN_DEFAULTS if kind=='lora' else FFT_DEFAULTS)
    st.caption('LoRA 默认值来自 V2 附录 D.3 / 表 11；FFT 为独立比较配置。')
    if kind=='lora':
        st.subheader('逐层 QKV rank')
        st.caption('每个投影 rank=0：完全冻结，不挂 LoRA。原模型基础权重始终冻结。')
        with st.container(horizontal=True):
            q=st.number_input('批量 Q',0,1024,4,key='bulk_q')
            k=st.number_input('批量 K',0,1024,0,key='bulk_k')
            v=st.number_input('批量 V',0,1024,4,key='bulk_v')
            if st.button('应用到全部层'):
                st.session_state['rank_rows']=[dict(layer=i,q=int(q),k=int(k),v=int(v)) for i in range(24)]
                st.session_state['rank_version']=st.session_state.get('rank_version',0)+1
            if st.button('全部冻结'):
                st.session_state['rank_rows']=[dict(layer=i,q=0,k=0,v=0) for i in range(24)]
                st.session_state['rank_version']=st.session_state.get('rank_version',0)+1
        rows=st.session_state.setdefault('rank_rows',[dict(layer=i,**r) for i,r in enumerate(config['ranks'])])
        table=st.data_editor(pd.DataFrame(rows),disabled=['layer'],hide_index=True,
            key=f"rank_editor_{st.session_state.get('rank_version',0)}",num_rows='fixed',
            column_config={x:st.column_config.NumberColumn(x.upper(),min_value=0,max_value=1024,step=1) for x in ['q','k','v']},
            alt='24 层 Q K V 的独立 rank')
        config['ranks']=[{key:int(row[key]) for key in ['q','k','v']} for row in table.to_dict('records')]
    with st.form('training_config'):
        a,b,c=st.columns(3)
        config['learning_rate']=a.number_input('学习率',min_value=1e-9,value=config['learning_rate'],format='%.7f')
        config['epochs']=b.number_input('Epochs',min_value=.1,value=float(config['epochs']))
        config['seed']=c.number_input('Seed',min_value=0,value=config['seed'])
        config['batch_size']=a.number_input('每设备 batch size',min_value=1,value=config['batch_size'])
        config['gradient_accumulation']=b.number_input('梯度累积',min_value=1,value=config['gradient_accumulation'])
        config['eval_batch_size']=c.number_input('验证 batch size',min_value=1,value=config['eval_batch_size'])
        config['warmup_steps']=a.number_input('Warmup steps',min_value=0,value=config['warmup_steps'])
        config['weight_decay']=b.number_input('Weight decay',min_value=0.,value=config['weight_decay'])
        config['label_smoothing']=c.number_input('Label smoothing',min_value=0.,max_value=.99,value=config['label_smoothing'])
        config['dropout']=a.number_input('Dropout',min_value=0.,max_value=.99,value=config['dropout'])
        if kind=='lora':
            config['alpha']=b.number_input('LoRA alpha',min_value=1,value=config['alpha'])
        config['precision']=c.selectbox('精度',['fp32','fp16','bf16'])
        config['gradient_checkpointing']=st.checkbox('梯度检查点',value=config['gradient_checkpointing'])
        config['max_length']=st.number_input('最大训练长度',min_value=32,max_value=1024,value=config['max_length'])
        config['logging_steps']=st.number_input('日志间隔（steps）',min_value=1,value=config['logging_steps'])
        with st.expander('少量样本 / 冒烟设置'):
            n=st.number_input('训练样本数，0=全部',min_value=0,value=0)
            e=st.number_input('验证样本数，0=全部',min_value=0,value=0)
            steps=st.number_input('最大 steps，0=按 epochs',min_value=0,value=0)
            config.update(train_samples=n or None,eval_samples=e or None,max_steps=steps or -1)
        submitted=st.form_submit_button('启动训练',type='primary')
    st.caption(f"有效 batch size：{config['batch_size']*config['gradient_accumulation']}。")
    duration(config,kind)
    if submitted:
        if kind=='lora' and not any(v for row in config['ranks'] for v in row.values()):
            st.error('全部投影已冻结，没有可训练参数；请启用至少一个 rank，或使用 Base 测试。')
        else:
            start([(kind,config)])

elif page=='测试 / benchmark':
    if st.button('刷新模型列表'):
        models.clear()
    options=models()
    selected=st.multiselect('选择模型（仅 medium 项目）',range(len(options)),default=[0],
        format_func=lambda i:f"{options[i]['type']} · {options[i]['name']}")
    for i in selected:
        tb_button(options[i],'eval_tb_'+str(i))
    config=deepcopy(EVAL_DEFAULTS)
    with st.form('evaluation_config'):
        kind=st.selectbox('任务',['benchmarks','tests'])
        config['split']=st.selectbox('数据 split',['validation','test'])
        config['metrics']=st.multiselect('评测指标',METRICS,default=METRICS)
        config['samples']=st.number_input('样本数，0=全部',min_value=0,value=0) or None
        config['max_new_tokens']=st.number_input('最大新 token 数',min_value=1,value=60)
        config['num_beams']=st.number_input('Beam size',min_value=1,value=10)
        config['length_penalty']=st.number_input('Length penalty',value=.9)
        config['no_repeat_ngram_size']=st.number_input('No repeat ngram size',min_value=0,value=4)
        submitted=st.form_submit_button('启动评测',type='primary')
    if selected:
        duration({**config,'model':options[selected[0]]},kind)
    if submitted:
        if not selected or not config['metrics']:
            st.error('请选择模型及至少一个指标。')
        else:
            start([(kind,{**deepcopy(config),'model':options[i]}) for i in selected])

elif page=='历史测试记录':
    st.subheader('历史测试记录')
    items=[r for r in records() if r['kind'] in ('tests','benchmarks')]
    model_type=st.selectbox('模型类型',['全部','base','lora','fft'])
    state=st.selectbox('状态',['全部','completed','failed','running','queued','interrupted'])
    text=st.text_input('按模型、运行日期或目录名筛选')
    metric=st.selectbox('评测项',['全部',*METRICS])
    items=[r for r in items if (state=='全部' or r['status']==state)
        and (model_type=='全部' or r['config'].get('model',{}).get('type')==model_type)
        and text.lower() in (r['path']+str(r['config'].get('model',{}))).lower()
        and (metric=='全部' or metric in r['config'].get('metrics',[]))]
    if not items:
        st.info('暂无匹配记录。')
    else:
        index=st.selectbox('测试运行',range(len(items)),format_func=lambda i:Path(items[i]['path']).name)
        record=items[index]
        path=Path(record['path'])
        st.json({k:v for k,v in record.items() if k!='config'})
        st.json(read_json(path/'metrics.json',{}))
        with st.expander('运行配置'):
            st.json(record['config'])
        tb_button(record['config'].get('model',{}),'history_tb')
        for filename in ['predictions.jsonl','sample_scores.jsonl','metrics.json','metrics.csv','config.json','run.log','gpu.jsonl']:
            file=path/filename
            if file.exists():
                st.download_button('下载 '+filename,file.read_bytes(),file_name=filename,key='download_'+filename)
        scores=path/'sample_scores.jsonl'
        if scores.exists():
            frame=pd.read_json(scores,lines=True)
            st.dataframe(frame,alt='逐样本预测、参考答案及测试成绩')
        if (path/'predictions.jsonl').exists():
            rescore=st.multiselect('重新评分指标',METRICS,default=record['config'].get('metrics',METRICS))
            if st.button('使用已保存预测重新评分'):
                config={**record['config'],'predictions':str(path/'predictions.jsonl'),'metrics':rescore}
                if rescore:
                    start([('benchmarks',config)])
                else:
                    st.error('至少选择一个指标。')

else:
    st.subheader('任务进度')
    @st.fragment(run_every=3)
    def progress_panel():
        items=history()
        if not items:
            st.info('暂无任务。')
            return
        chosen=st.selectbox('运行',range(len(items)),format_func=lambda i:Path(items[i]['path']).name,key='progress_run')
        record=items[chosen]
        path=Path(record['path'])
        st.caption(f"{record['kind']} · {record['status']}")
        if record['status']=='running' and not alive(record.get('pid')):
            st.warning('任务进程已退出，未写入完成状态，请查看 process.log。')
        state=read_json(path/'progress.json',{})
        if state.get('total'):
            st.progress(min(state['completed']/state['total'],1.),text=state.get('phase',''))
        seconds=state.get('remaining_seconds') if record['status']=='running' else None
        if seconds is None and record['status'] in ('queued','running'):
            average,count=estimate(record['config'],record['kind'])
            if average is not None:
                elapsed=(datetime.now(HK)-datetime.fromisoformat(record['started'])).total_seconds() if record.get('started') else 0
                seconds=max(0,average-elapsed)
                st.caption(f'依据 {count} 次同配置成功运行的耗时均值')
        if seconds is not None:
            eta=datetime.now(HK)+timedelta(seconds=seconds)
            st.metric('预计完成时间',eta.strftime('%H:%M:%S'))
        elif record['status'] in ('queued','running'):
            st.caption('暂无历史估算，等待运行进度。')
        if record.get('elapsed_seconds') is not None:
            st.metric('总耗时（秒）',f"{record['elapsed_seconds']:.1f}")
        for filename in ['run.log','process.log']:
            if (path/filename).exists():
                with (path/filename).open('rb') as stream:
                    stream.seek(max(0,(path/filename).stat().st_size-16000))
                    content=stream.read().decode('utf-8',errors='replace')
                st.code(content,language='text')
        if (path/'gpu.jsonl').exists():
            try:
                gpu=pd.read_json(path/'gpu.jsonl',lines=True)
                st.dataframe(gpu.tail(20),alt='GPU 占用率、核心温度及显存采样')
            except ValueError:
                st.caption('GPU 采样正在写入。')
    progress_panel()
