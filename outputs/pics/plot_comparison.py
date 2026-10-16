"""汇总正式 benchmark，生成每项指标的五组横向对比图。"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FONT = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams['font.family'] = FONT.get_name()
plt.rcParams['axes.unicode_minus'] = False
METRICS = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'NIST', 'METEOR', 'ROUGE-L', 'CIDEr']
records = []
for run in sorted((ROOT / 'outputs/benchmarks').iterdir()):
    if not run.is_dir():
        continue
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    result = json.loads((run / 'metrics.json').read_text(encoding='utf-8'))
    status = json.loads((run / 'status.json').read_text(encoding='utf-8'))
    assert status['status'] == 'completed' and not result['errors']
    assert result['samples'] == 1847 and result['profile'] == 'small-compatible single-target'
    assert not config['smoke']
    records.append(dict(run=str(run.relative_to(ROOT)), config=config, result=result,
                        kind=config['model']['type'],
                        seed=config['model'].get('training_config', {}).get('seed')))
keys = ['dataset', 'split', 'samples', 'max_new_tokens', 'num_beams', 'length_penalty',
        'no_repeat_ngram_size', 'seed', 'local_data']
assert all(all(r['config'][k] == records[0]['config'][k] for k in keys) for r in records)
groups = [
    ('原生 GPT-2 Medium', [r for r in records if r['kind'] == 'base']),
    ('LoRA · 3 次均值', [r for r in records if r['kind'] == 'lora' and r['seed'] in [42,43,44]]),
    ('LoRA · 5 次均值', [r for r in records if r['kind'] == 'lora']),
    ('FFT · 3 次均值', [r for r in records if r['kind'] == 'fft' and r['seed'] in [42,43,44]]),
    ('FFT · 5 次均值', [r for r in records if r['kind'] == 'fft']),
]
assert [len(rs) for _, rs in groups] == [1,3,5,3,5]
assert all(sorted(r['seed'] for r in groups[i][1]) == [42,43,44] for i in [1,3])
colors = ['#8B98A9', '#6BAED6', '#2474AE', '#EDA86B', '#D7732C']
rows = []
for metric in METRICS:
    values = [float(np.mean([r['result']['metrics'][metric] for r in rs])) for _, rs in groups]
    fig, ax = plt.subplots(figsize=(11, 5.8), facecolor='#F7F9FC')
    ax.set_facecolor('#F7F9FC')
    bars = ax.barh(np.arange(5), values, color=colors, height=.57)
    ax.invert_yaxis()
    ax.set_yticks(np.arange(5), [label for label, _ in groups], fontsize=12)
    ax.set_xlim(0, max(values)*1.22)
    ax.xaxis.grid(True, color='#E1E6ED', linewidth=.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='both', length=0, colors='#344054')
    ax.set_xlabel('原始分数（越高越好）', labelpad=12, color='#475467')
    for bar, value in zip(bars, values):
        ax.text(value+max(values)*.015, bar.get_y()+bar.get_height()/2,
                f'{value:.4f}', va='center', fontsize=12, fontweight='bold', color='#26354A')
    fig.text(.04, .93, f'{metric}  |  GPT-2 Medium 横向对比', fontsize=20, weight='bold', color='#172B4D')
    fig.text(.04, .875, 'E2E test · 1,847 条 · 单 target 评分 · beam=10 · max new tokens=64', fontsize=10, color='#667085')
    fig.text(.04, .075, '3 次：seed 42 / 43 / 44；5 次：各方法全部 5 个模型的算术均值（包含前述 3 次）。', fontsize=9, color='#667085')
    fig.text(.04, .035, 'LoRA 全部 seed：42, 43, 44, 110, 110；FFT：42, 43, 44, 101, 101。分数未乘 100。', fontsize=9, color='#667085')
    fig.subplots_adjust(left=.23, right=.96, bottom=.22, top=.8)
    fig.savefig(OUT / f'{metric}.png', dpi=220)
    fig.savefig(OUT / f'{metric}.svg')
    plt.close(fig)
    rows.append(dict(metric=metric, **{name: value for (name,_),value in zip(groups,values)}))
with (OUT/'comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(OUT/'sources.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(rows, ensure_ascii=False, indent=2))
