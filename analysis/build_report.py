"""Build the Markdown report and static plots from audited benchmark records."""
import json
from pathlib import Path
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

OUT=Path(__file__).resolve().parent
R=json.loads((OUT/'audit_results.json').read_text(encoding='utf8'))
D=json.loads((OUT/'data_audit.json').read_text(encoding='utf8'))
L=json.loads((OUT/'length_audit.json').read_text(encoding='utf8'))
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,
                     'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
MET=['BLEU-4','NIST','METEOR','ROUGE-L','CIDEr']
ALL=['BLEU-1','BLEU-2','BLEU-3']+MET
SCALE={k:100 if k.startswith('BLEU') or k in ['METEOR','ROUGE-L'] else 1 for k in ALL}
PAPER={'lora':[70.4,8.85,46.8,71.8,2.53], 'fft':[68.2,8.62,46.2,71.0,2.47]}
COL={'lora':'#2563a6','fft':'#dc7b30'}


def group(kind,n=3):
    """Select the requested three-seed group or all five records."""
    return [r for r in R if r['kind']==kind and (n==5 or r['seed'] in [42,43,44])]


def mean(kind,k,n=3):
    """Return the mean in the report's display scale."""
    return st.mean(r['metrics'][k]*SCALE[k] for r in group(kind,n))


def resc(kind,k,profile):
    """Return controlled rescoring mean for seeds 42, 43 and 44."""
    return st.mean(r['rescoring'][profile][k] for r in group(kind))


def table(headers, rows):
    """Format a flat Markdown table."""
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+
                     ['|'+'|'.join(map(str,row))+'|' for row in rows])


def save(name):
    """Save both a PNG preview and a vector version."""
    plt.savefig(FIG/(name+'.png'),bbox_inches='tight')
    plt.savefig(FIG/(name+'.svg'),bbox_inches='tight')
    svg=FIG/(name+'.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf8').splitlines())+'\n',encoding='utf8')
    plt.close()


fig,axes=plt.subplots(1,5,figsize=(16,4.6),layout='constrained')
for i,(k,ax) in enumerate(zip(MET,axes)):
    for j,kind in enumerate(['lora','fft']):
        ax.bar(j-.18,PAPER[kind][i],width=.34,color=COL[kind],alpha=.35)
        ax.bar(j+.18,mean(kind,k),width=.34,color=COL[kind])
        ax.scatter(j+.18,mean(kind,k,5),marker='D',s=25,c='black',zorder=4)
        ax.text(j-.18,PAPER[kind][i],f'{PAPER[kind][i]:.2f}',ha='center',va='bottom',fontsize=8)
        ax.text(j+.18,mean(kind,k),f'{mean(kind,k):.2f}',ha='center',va='bottom',fontsize=8)
    ax.set_xticks([0,1],['LoRA','FFT']);ax.set_title(k)
    ax.set_ylim(0,max(PAPER['lora'][i],PAPER['fft'][i],mean('lora',k))*1.2)
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('原文与本次分数：口径未统一的表面差距',fontsize=17)
fig.supxlabel('浅色柱：论文；深色柱：本次 3 次均值；黑菱形：本次 5 次均值。METEOR 实现不同，分数更高不代表超越论文。',fontsize=10)
save('paper_vs_local')

fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
for ax,title,key1,key2,scale,pidx in [(axes[0],'BLEU-4 / BLEU', 'COCO_BLEU4','MTEval_BLEU',100,0),
                                     (axes[1],'NIST','NLTK_NIST','MTEval_NIST',1,1)]:
    for kind in ['lora','fft']:
        vals=[resc(kind,key1,'single'),resc(kind,key1,'multi'),resc(kind,key2,'single'),resc(kind,key2,'multi')]
        vals=np.array(vals)*scale
        offset=-.07 if kind=='lora' else .07
        ax.plot(np.arange(4)+offset,vals,'o-',color=COL[kind],label=f'{kind.upper()} 三次均值')
        ax.axhline(PAPER[kind][pidx],ls='--',color=COL[kind],alpha=.55,label=f'论文 {kind.upper()}')
    ax.set_xticks(range(4),['项目单参考','项目实现多参考','MTEval 单参考','MTEval 多参考'],rotation=12)
    ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=9)
fig.suptitle('固定预测文本的评分对照：多参考与评分实现均能改变分数',fontsize=16)
fig.supxlabel('4 个条件均使用同一批 GEM 输入及预测；MTEval 多参考仍不是论文原始测试集。',fontsize=10)
save('controlled_rescoring')

fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
for k,ax in zip(['BLEU-4','ROUGE-L','CIDEr'],axes):
    for kind in ['lora','fft']:
        rr=sorted(group(kind),key=lambda r:r['seed'])
        ax.plot([r['seed'] for r in rr],[r['metrics'][k]*SCALE[k] for r in rr], 'o-',color=COL[kind],label=kind.upper())
    ax.set_xticks([42,43,44]);ax.set_xlabel('训练 seed');ax.set_title(k);ax.grid(alpha=.2);ax.legend()
fig.suptitle('统一标准的三次实验：LoRA / FFT 的相对表现随指标和 seed 变化',fontsize=16)
save('seed_comparison')

paper_rows=[]
for kind in ['lora','fft']:
    for i,k in enumerate(MET):
        p=PAPER[kind][i];a=mean(kind,k);b=mean(kind,k,5)
        paper_rows.append([kind.upper(),k,f'{p:.3f}',f'{a:.3f}',f'{a-p:+.3f}',f'{b:.3f}',f'{b-p:+.3f}'])

mean_rows=[]
for k in ALL:
    a,b=mean('lora',k),mean('lora',k,5)
    c,d=mean('fft',k),mean('fft',k,5)
    mean_rows.append([k,f'{a:.4f}',f'{b:.4f}',f'{b-a:+.4f}',f'{c:.4f}',f'{d:.4f}',f'{d-c:+.4f}'])

sd_rows=[]
for k in MET:
    sd_rows.append([k]+[f"{mean(kind,k):.4f} ± {st.stdev(r['metrics'][k]*SCALE[k] for r in group(kind)):.4f}" for kind in ['lora','fft']])

relative=[]
for i,k in enumerate(MET):
    delta=mean('lora',k)-mean('fft',k)
    diff=[next(r for r in group('lora') if r['seed']==seed)['metrics'][k]*SCALE[k] -
          next(r for r in group('fft') if r['seed']==seed)['metrics'][k]*SCALE[k] for seed in [42,43,44]]
    relative.append([k,f"{PAPER['lora'][i]-PAPER['fft'][i]:+.4f}",f'{delta:+.4f}',
                     f"{mean('lora',k,5)-mean('fft',k,5):+.4f}",', '.join(f'{d:+.4f}' for d in diff)])

control=[]
for kind in ['lora','fft']:
    for name,key,scale in [('BLEU-4（COCO）','COCO_BLEU4',100),('NIST（NLTK）','NLTK_NIST',1),
                           ('ROUGE-L（未加 PTB）','ROUGE-L',100),('CIDEr（未加 PTB）','CIDEr',1),
                           ('BLEU（MTEval）','MTEval_BLEU',100),('NIST（MTEval）','MTEval_NIST',1)]:
        a,b=resc(kind,key,'single')*scale,resc(kind,key,'multi')*scale
        control.append([kind.upper(),name,f'{a:.4f}',f'{b:.4f}',f'{b-a:+.4f}'])

run_rows=[]
for r in R:
    run_rows.append([r['label']]+[f"{r['metrics'][k]*SCALE[k]:.4f}" for k in MET])

length_rows=[]
for kind,seed in [('lora',110),('fft',101)]:
    rr=sorted([r for r in R if r['kind']==kind and r['seed']==seed],key=lambda r:r['max_length'])
    for k in MET:
        a,b=[r['metrics'][k]*SCALE[k] for r in rr]
        length_rows.append([kind.upper(),seed,k,f'{a:.5f}',f'{b:.5f}',f'{b-a:+.5f}'])

lo_b=resc('lora','MTEval_BLEU','multi')*100
lo_n=resc('lora','MTEval_NIST','multi')
ft_b=resc('fft','MTEval_BLEU','multi')*100
ft_n=resc('fft','MTEval_NIST','multi')

text=f'''# GPT-2 Medium / E2E：与 LoRA 原文的数据一致性及差异原因报告

日期：2026-10-17。结果来自当前仓库 `outputs/benchmarks` 的 11 个正式 benchmark；代码基线为 `117a9d1`。用户确认当前代码与家庭电脑实际运行代码一致。本报告不改写原始结果，也未重新训练模型。

## 1. 结论

**本次结果与原文的绝对分数不一致，目前不能称为完成了原文数值复现。已有证据首先指向数据集与评分口径不同，不能据此判定模型训练失败。**

- 原始三次均值中，LoRA 的 BLEU-4 为 **{mean('lora','BLEU-4'):.3f}**，论文为 **70.4**；NIST 为 **{mean('lora','NIST'):.3f}**，论文为 **8.85**。这不是单纯把 0–1 分数乘 100 就能消除的差距。
- 固定模型预测、把参考文本由第一条改为全部，再换成官方流程使用的 Python MTEval 评分代码，LoRA 的 BLEU 变为 **{lo_b:.3f}**、NIST 变为 **{lo_n:.3f}**。因此，**评分协议确实解释了一部分差距**；仍有 **{70.4-lo_b:.3f} BLEU / {8.85-lo_n:.3f} NIST** 的表面差值未消除。这部分尚不能直接解释为模型能力差距，因为测试输入集合仍不同。
- **相对趋势只部分一致**：原始三次均值下，LoRA 的 BLEU-4、NIST、ROUGE-L、CIDEr 略高于 FFT，METEOR 略低；BLEU-4 的领先仅 **{mean('lora','BLEU-4')-mean('fft','BLEU-4'):.4f}**，不能写成稳定、全面优于 FFT。
- **3 次与 5 次均值会改变“领先多少”的印象，但不改变当前不具备严格论文可比性的判断。**主分析采用 seed 42/43/44，5 次均值按用户要求完整保留，并在第 4 节单独解释其影响。
- **512/1024 没有造成训练或验证样本截断差异**：实际编码长度上限分别为 124 / 135 tokens。LoRA 两条长度对照记录预测完全一致；FFT 有少量分数差异，现有证据不能将其归因于长度截断。

## 2. 实验分组与比较规则

|用途|LoRA|FFT|解释|
|---|---|---|---|
|主要三次均值|seed 42/43/44，max_length=512|seed 42/43/44，max_length=1024|各方法内部配置统一；两种方法保留各自标准设置|
|五次均值|三次组 + seed 110 的 512/1024 两条记录|三次组 + seed 101 的 512/1024 两条记录|全部五个记录的算术均值，不等于五个独立种子|
|长度对照|seed 110：512 对 1024|seed 101：512 对 1024|固定各方法的原始种子，观察训练长度变化|
|未微调基线|GPT-2 Medium Base 一次|同一个 Base|仅说明任务适配效果，原文表 3 无对应未微调行|

训练 seed 来自 `config.json → model.training_config.seed`；所有 benchmark 外层生成配置的 `seed=42`，不能把它误当成所有模型的训练 seed。全部运行均为 test、1,847 条、beam=10、max_new_tokens=64、length_penalty=0.9、no_repeat_ngram_size=4。预测采用确定性 beam search。

展示尺度：BLEU-1～4、METEOR、ROUGE-L 均为原始分数 ×100；NIST、CIDEr 保留原始尺度。论文 BLEU 对照本次 BLEU-4；论文没有给出该表的 BLEU-1～3。下面的差值表示“本次 − 论文”，是口径尚未统一时的描述性差值，不能用作同条件效果检验。

## 3. 与原文直接对照：绝对数值不一致

原文参照 **LoRA v2 表 3 的 GPT-2 Medium 行**，不混用 GPT-2 Large、Small 或其他数据集。原文 FT 行带星号，来自 Li & Liang (2021)，不是 LoRA 作者在同一表中重新运行的 FT 结果。[原文表 3 与 D.3](https://arxiv.org/html/2106.09685#S5.T3)

{table(['方法','指标','论文','本次 3 次均值','3 次差值','本次 5 次均值','5 次差值'],paper_rows)}

![原文和本次 3/5 次均值对照](figures/paper_vs_local.png)

这里 METEOR 的反向差异尤其重要：本次得分比论文高，不能解释为模型超越原文。项目使用 NLTK METEOR 的逐句平均，官方 E2E 流程使用 COCO 包装的 METEOR 实现；匹配、归一化和汇总协议不同。只做 ×100 并没有统一算法。

本次三次实验的离散程度如下，`±` 为 **样本标准差（ddof=1）**。论文表 3 将自己的区间称为 confidence intervals，不能直接把本表标准差与原文区间当作同一种统计量。[原文说明](https://arxiv.org/html/2106.09685#S5.T3)

{table(['指标','LoRA：均值 ± 样本标准差','FFT：均值 ± 样本标准差'],sd_rows)}

## 4. 3 次和 5 次均值：是否影响结论

{table(['指标','LoRA 3 次','LoRA 5 次','5−3','FFT 3 次','FFT 5 次','5−3'],mean_rows)}

**有影响，但影响主要是统计混合方式，而非增加两个独立种子的证据。**

LoRA 加入的两条 seed=110 记录，1,847 条预测逐条完全相同；因此五次均值对这组相同预测赋予 2/5 权重。它们属于有意设置的同 seed 长度对照，五次均值描述的是五份实验记录，而不是五个独立 seed。FFT 新增两条 seed=101 记录，其 BLEU-1～4、NIST、METEOR、CIDEr 低于 seed 42/43/44 的平均值，使这些指标的五次均值下降；ROUGE-L 则小幅上升。五次均值同时改变种子构成与长度构成，不能将 5−3 全归因为某一个因素。

|需要说明的变化|三次均值|五次均值|含义|
|---|---|---|---|
|LoRA−FFT 的 BLEU-4|{mean('lora','BLEU-4')-mean('fft','BLEU-4'):+.4f}|{mean('lora','BLEU-4',5)-mean('fft','BLEU-4',5):+.4f}|五次混合后，视觉上会放大 LoRA 的领先幅度|
|LoRA−FFT 的 METEOR|{mean('lora','METEOR')-mean('fft','METEOR'):+.4f}|{mean('lora','METEOR',5)-mean('fft','METEOR',5):+.4f}|正负号发生改变；“所有指标都领先”依赖分组方式|

**报告引用规则：主要结果引用三次均值，五次均值作为补充描述；涉及长度影响时，只引用第 7 节的同 seed 对照。**

## 5. 差异原因一：评分时只使用了第一条参考——已做受控验证

当前 [`src/utils/scoring.py`](../src/utils/scoring.py) 的 `score()` 明确构造 `gts = {{i: [row["target"]] ...}}`。BLEU、NIST、METEOR、ROUGE-L、CIDEr 都没有使用保存的全部 `references`。数据加载器把 `target` 设为第一条参考，记录也逐条满足 `target == references[0]`。

完整性核验得到：

|审计项|结果|
|---|---|
|正式 benchmark 数量|11，全部 completed，无评分 errors|
|每个运行的预测条数|1,847，ID/MR/target/references 与其他运行逐条相同|
|参考总数|4,693|
|只有一条参考的输入|828 / 1,847（44.83%）|
|有多条参考的输入|1,019 / 1,847（55.17%）|
|每个输入参考数|平均 {4693/1847:.3f}，最多 35|

以下对照**完全固定已有预测**，只切换第一条参考/全部参考。所有分数取 seed 42/43/44 三次均值。COCO 与 NLTK 的“单参考”列重新计算后，与原 benchmark 对应分数的绝对误差均小于 `1e-10`；共核对 11×4=44 个分数。因此没有发现这四项原始分数的汇总或抄录错误。

{table(['方法','指标实现','第一条参考','全部参考','增加量'],control)}

![固定预测的参考数和评分实现对照](figures/controlled_rescoring.png)

两种作用可以分开看：

1. **参考集合的作用**：比较同一个实现内的“第一条 / 全部”两列，可以确认丢弃参考文本使这些记录的得分下降。
2. **评分实现的作用**：在同一个参考条件下，比较项目的 COCO BLEU / NLTK NIST 与 MTEval 的 BLEU / NIST，可以确认分词、大小写与算法实现也会改变数值。项目直接传入原始字符串/空白分词，MTEval 自带文本处理。

例如 LoRA 的 BLEU 从原始 {mean('lora','BLEU-4'):.3f} 到 MTEval 多参考 {lo_b:.3f}，增加 {lo_b-mean('lora','BLEU-4'):.3f}，约占原始表面 BLEU 差值的 {(lo_b-mean('lora','BLEU-4'))/(70.4-mean('lora','BLEU-4'))*100:.1f}%。这只是本次改变评分协议后的算术分解，**不是对论文差距的完整因果解释**。该重评分结果仍是 GEM 任务上的诊断结果。

**相对排名也受协议影响**：原始单参考 NLTK NIST 中，LoRA−FFT 为 +0.0298；改成 MTEval 多参考后变为 -0.0237（LoRA 6.3302，FFT 6.3539）。因此，评分差异不仅改变绝对分数，也可能改变某一指标上的优劣次序，不能只用原始三次均值宣称复现了原文的全面领先。

本次没有运行完整官方 COCO/PTB/Java METEOR 流程：当前分析环境 PATH 中未找到 Java。上表 ROUGE-L/CIDEr 仅改变参考集合，仍未经过官方 PTB 流程；METEOR 没有新增“官方复现分数”。不能把这份重评分表命名为“官方完整评测结果”。[官方 E2E 评分入口](https://github.com/tuetschek/e2e-metrics/blob/master/measure_scores.py)

## 6. 差异原因二：数据集内容与 MR 分组不同——已核对源文件

本项目使用 `GEM/e2e_nlg`，其加载器训练端读取清洗后的 `train-fixed.no-ol.csv`，测试端读取 GEM 分组后的 JSON；LoRA 官方流程直接处理仓库中的 `data/e2e/train.txt`、`test.txt`。这不是两个名称指向相同文件。[GEM 加载器](https://huggingface.co/datasets/GEM/e2e_nlg/blob/main/e2e_nlg.py)、[LoRA 数据处理入口](https://github.com/microsoft/LoRA/blob/main/examples/NLG/create_datasets.sh)

|项目|本次 GEM 数据|LoRA 官方公开数据快照|意义|
|---|---|---|---|
|训练样本/参考行数|33,525|42,061|相同 5 epochs 的样本量、更新步数及训练分布均不同|
|测试输入 MR 数|1,847|630|测试任务的输入集合与参考分组不同|
|测试参考总数|4,693|4,693|总参考数相同不等于任务相同|
|每个测试输入平均参考数|{4693/1847:.3f}|{4693/630:.3f}|GEM 分组较细，参考匹配机会也不同|

这些数量来自下载源文件的实际计数；GEM test 中 MR、参考数组与本次全部预测记录逐条一致。官方测试的 4,693 行按官方 context 分组后是 630 个输入，而当前 GEM 记录本身就是 1,847 个不同 MR，**不是“把 4,693 条测试漏跑成 1,847 条”**。GEM 数据卡明确说明其清洗包含 MR 修正与重叠清除；这可能把原来同一输入下的参考分到不同输入。当前对照能确认任务已经变化，但不能给这类变化分配一个独立的性能损失百分比。[GEM 数据结构与清洗说明](https://huggingface.co/datasets/GEM/e2e_nlg#dataset-structure)

训练样本减少 8,536 条，约 20.29%；LoRA 每 epoch 从官方约 5,258 个 batch 变为当前约 4,191 个 batch（batch=8，尾 batch 处理会影响精确值），相同的 warmup=500 对应不同训练比例。这里数量不同是事实，对最终分数的独立贡献尚需统一训练数据重跑才能确定。

**因此，即使把当前预测改为多参考评分，仍不能直接与原文 70.4 BLEU 认定为同条件复现。**本次 MTEval 多参考的 LoRA / FFT 分别为 {lo_b:.3f} / {ft_b:.3f} BLEU、{lo_n:.3f} / {ft_n:.3f} NIST；残余差异也不能全部归到优化器或 seed 上。

## 7. 长度与 seed 对照：哪些确实产生影响

### 7.1 相同 seed 下的 512 / 1024

{table(['方法','固定 seed','指标','max_length=512','max_length=1024','1024−512'],length_rows)}

|核验项|LoRA seed=110|FFT seed=101|
|---|---|---|
|两种长度的预测完全相同|1,847 / 1,847|1,598 / 1,847|
|预测文本不同|0|249（13.48%）|
|能否据此认为 1024 提升有效输入信息|不能|不能|

这里用缓存的 GPT-2 Medium tokenizer，严格按 [`tokenize()`](../src/utils/data.py) 的方式分别编码提示词与带前导空格的 target，再加 EOS，检查了公开源文件的全部训练/验证样本：

|数据|样本数|未截断时最大 tokens|超过 512|超过 1024|
|---|---|---|---|---|
|train|33,525|124|0|0|
|validation（每 MR 第一条参考）|1,484|135|0|0|

代码采用动态 padding，`max_length` 用于切片，不会把每条样本强行补到该长度。因此在这份数据和代码下，512 改为 1024 **没有改变进入模型的训练/验证 token 序列**。这很好地解释了 LoRA 长度对照的相同预测，也排除了“FFT 因 512 截断而变差”这一解释。

FFT 的 BLEU-4 差 {abs(float(length_rows[5][-1])):.5f}（展示尺度），其他指标差值也很小。相同 seed 并不自动保证 GPU 训练逐位相同，当前代码没有显式启用严格确定性算法；数值扰动、运行状态、权重保存/加载等可造成不同预测。但这只是后续核验方向，**不能仅凭现有 benchmark 指定唯一原因**。训练权重、家庭电脑的完整 Trainer 状态和逐步日志尚未在本报告中逐一匹配，不能据此断言权重相同或不同。

长度统计依据下载的当前 GEM 数据快照和本地 GPT-2 tokenizer；来源哈希见 `sources_manifest.json` / `length_audit.json`。这提供了有力的无截断证据，但原始训练记录若实际使用其他数据修订版，还需以当次训练缓存/指纹作最终核验。

### 7.2 seed 42/43/44 的变化与相对趋势

![不同 seed 的 BLEU、ROUGE-L、CIDEr](figures/seed_comparison.png)

下表全部为 LoRA−FFT；同 seed 列按 42 / 43 / 44 排列，仍保留各方法自己的标准训练设置。

{table(['指标','论文 LoRA−FT','本次 3 次均值差','本次 5 次均值差','本次同 seed 差（42 / 43 / 44）'],relative)}

BLEU-4 在 seed=43 上出现 LoRA 低于 FFT，说明三次平均的微小正差并非每个种子都成立。三次组中 ROUGE-L 每个 seed 都由 LoRA 领先；其他指标仍有波动。样本只有三个训练种子，没有进行同协议的大规模重复或显著性检验，应写为“当前均值接近、部分指标略优”，不能写为稳定压倒 FFT。

## 8. 训练与解码设置：已对齐部分与仍存差异

|检查项|当前代码 / 记录|原文 / 官方来源|判断|
|---|---|---|---|
|LoRA 学习率、epoch、batch、warmup|2e-4 / 5 / 8 / 500|原文表 11 相同|已对齐|
|LoRA Q/V rank、alpha、weight decay、label smoothing|4 / 32 / 0.01 / 0.1|原文表 11 相同|已对齐|
|LoRA dropout|adapter=0.1，基础 GPT-2 dropout=0|官方实现对应结构|不是把 FFT 的基础 dropout 直接套给 LoRA|
|FFT 的学习率、epoch、batch|5e-5 / 5 / 10|项目按 Prefix-Tuning FT 基线配置|大项对齐不代表实现完全相同|
|提示词|`MR: …` + 换行 + `Text:`|官方输入为 `name : …`、`Type : …` 等竖线分隔的 context 格式|文本格式确有不同，独立影响未量化|
|beam / 最大生成长度 / 禁重复 ngram|10 / 64 / 4|官方示例相同|不能首先归因为 beam 或生成上限设置错|
|length penalty|0.9|论文表 11=0.9；当前官方 README 示例=0.8|项目对齐论文表，不等于逐字复刻 README 命令|
|生成终止 / 文本清理|HF `generate()` 默认模型 EOS，strip 后评分|官方示例 EOS=628，解码再按双换行截断|确有流程差异，未做固定模型生成消融|
|梯度裁剪|HF TrainingArguments 未覆盖 max_grad_norm，使用框架默认|官方示例 `--clip 0.0`|存在实现差异；不能量化为主因|
|最佳模型选择|每 epoch 验证，以 eval_loss 取最佳|原文 D.3 说明最佳 epoch|大策略接近；当次最佳 checkpoint 尚未逐一验证|

本段主要证据来自当前 `src/utils/training.py`、`data.py`、`evaluation.py` 及保存的 `training_config`；官方对照见 [NLG 复现命令](https://github.com/microsoft/LoRA/tree/main/examples/NLG#replicating-our-result-on-e2e)、[官方解码代码](https://github.com/microsoft/LoRA/blob/main/examples/NLG/src/gpt2_decode.py)。表 11 与 README 的 length penalty 本身不完全一致，报告保留这种源间差异，不将当前 0.9 直接判为错误。

## 9. 原因判断与最省成本的下一步

|候选原因|当前证据强度|已经知道什么|仍需什么才能闭环|
|---|---|---|---|
|单参考替代多参考|高，已有固定预测重评分|分数随参考集合明显改变|使用论文原始 MR/参考分组评分|
|评分库/分词/归一化不同|高，BLEU/NIST 已作同参考对照|算法差异解释部分分差；METEOR 原分不可直接对标|完整官方 E2E 包、PTB 与 Java METEOR|
|GEM 清洗数据替代原始数据|高，源文件数量和结构不同|训练样本、测试输入与分组均变化|统一原始数据后评测；必要时重新训练|
|seed 构成及 3/5 均值混合|高，已有全部运行配置|会改变领先幅度与 METEOR 正负关系|主结论固定三次组，长度组分开|
|512/1024 的截断差异|当前数据证据不支持|两种上限都未截断训练/验证样本|保留当次数据/分词器哈希后进一步核实|
|提示词、EOS、beam 实现、优化器细节|确有部分实现差异，贡献未证实|不能由配置大项相同推导出严格复现|固定权重单因素生成对照；之后再考虑训练消融|
|训练未完成、权重损坏、GPU 型号导致分数差距|目前证据不足|benchmark 均成功，不能据分差直接判定这类根因|当次训练状态、checkpoint、权重与加载记录|

建议按以下顺序补实验，避免过早重跑全部模型：

1. 保存本报告和现有原始 benchmark，保留“项目单参考”历史分数。若将来新增官方评分，使用独立 profile 和目录。
2. 在现有已训练权重上，对 LoRA 官方的 630 个原始 test 输入重新生成，使用官方 context 格式对应的方案，并用完整官方 E2E 脚本评分；这能优先分清评测协议与模型训练差异。提示词格式改变可能带来分布变化，应同时保留模型训练时使用的格式作为生成对照，不能直接覆盖。
3. 如果统一测试集和评分流程后仍有明显差距，再将训练集也换成官方 42,061 行版本，先用一个 seed 验证全链路；验证后再完成 42/43/44 三次对照。统一训练数据前，不能把残差归结为 rank、学习率或 epoch 不足。
4. FFT 长度对照可先在同一数据、同一初始化权重、严格确定性设置下做重复运行，记录输入 token 哈希和模型权重哈希；因为已无截断差异，这一步用于定位可重复性，而非验证“更长上下文”的收益。

这些是后续建议，本次没有改动训练、生成或评分主程序，没有启动上述新增实验。

## 10. 完整单次结果与可追溯附件

{table(['运行标签','BLEU-4','NIST','METEOR','ROUGE-L','CIDEr'],run_rows)}

|附件|内容|
|---|---|
|[run_metrics.csv](run_metrics.csv)|全部 11 次原始指标及运行目录；分数保持原始尺度|
|[rescoring.csv](rescoring.csv)|每次运行、每种参考条件的受控重评分；分数保持原始尺度|
|[audit_results.json](audit_results.json)|配置、原始分数、重评分、输入文件哈希、预测文本哈希|
|[data_audit.json](data_audit.json)|样本数量、MR 数量、参考分布核验|
|[length_audit.json](length_audit.json)|编码长度、tokenizer 哈希、预测一致条数|
|[sources_manifest.json](sources_manifest.json)|下载的官方/GEM 源文件 URL、大小、SHA-256|
|[analyze_consistency.py](analyze_consistency.py)|只读原 benchmark，重新核算分数及数据结构|
|[length_audit.py](length_audit.py)|独立核验编码长度和长度组预测|
|[build_report.py](build_report.py)|从审计结果生成 Markdown、PNG 与 SVG|
|[prepare_sources.py](prepare_sources.py)|下载并核验 manifest 中的源文件|
|[validation.json](validation.json)|分数复算、源文件哈希、表格/图片/链接与历史保留检查结果|

本次评分环境的 pycocoevalcap=1.2，与原 benchmark 保存版本一致；NIST 重新计算吻合。原始记录还保存 torch=2.14.0+cu130、transformers=5.17.0、datasets=2.21.0、peft=0.21.0、nltk=3.10.3。上述版本来自当次 `environment.json`，不等于本地分析环境版本。本地分析使用 Python 3.14.6、NLTK 3.10.0；与当次 NLTK 版本不同，但 NIST 原分复算吻合，未将本地 METEOR 当作原始或官方 METEOR 重算结果。

审计临时依赖和下载快照位于 `outputs/smoke/report_audit`，不混入正式 benchmark。运行审计需要 numpy、nltk、matplotlib、future、pycocoevalcap=1.2，以及长度统计使用的 tokenizers；依赖与外部源码保持独立。后续复跑时可运行 `python medium/prepare_sources.py` 获取 manifest 内文件（校验哈希），再运行 `analyze_consistency.py`、`length_audit.py`、`build_report.py`。

公开源码/数据快照读取于 2026-10-17；manifest 固定其内容哈希，但它们不构成论文作者当年运行环境的完整封存。本报告中**已证实的协议差异**与**未归因的剩余差距**已分别标注，后者不应写成已经查明的唯一训练根因。
'''
(OUT/'GPT2_Medium_数据一致性分析报告.md').write_text(text,encoding='utf8')
print('Report and three figure pairs generated.')
