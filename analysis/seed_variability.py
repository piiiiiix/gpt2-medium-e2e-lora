"""从已审计的正式结果生成均值、样本标准差、CV 与逐运行散点。"""
import csv
import hashlib
import json
from pathlib import Path
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
METRICS = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'NIST', 'METEOR', 'ROUGE-L', 'CIDEr']
COLORS = ['#94a3b8', '#60a5fa', '#2563eb', '#f7b267', '#dd792b']
MARKERS = {42: 'o', 43: 's', 44: '^', 110: 'D', 101: 'D', None: 'o'}


def summarize(records):
    """区分单次 Base、统一三 seed 组及五记录混合组，保留原始尺度。"""
    groups = [('Base', [r for r in records if r['kind'] == 'base'])]
    for kind, label in [('lora', 'LoRA'), ('fft', 'FT')]:
        selected = [r for r in records if r['kind'] == kind]
        groups.extend([(f'{label} 3', [r for r in selected if r['seed'] in [42, 43, 44]]),
                       (f'{label} 5', selected)])
    assert [len(rs) for _, rs in groups] == [1, 3, 5, 3, 5]
    assert all(sorted(r['seed'] for r in groups[i][1]) == [42, 43, 44] for i in [1, 3])
    rows = []
    points = []
    for metric in METRICS:
        for label, runs in groups:
            values = [r['metrics'][metric] for r in runs]
            avg = st.mean(values)
            sd = st.stdev(values) if len(values) > 1 else None
            rows.append(dict(metric=metric, group=label, n=len(values),
                             unique_seeds=len({r['seed'] for r in runs if r['seed'] is not None}),
                             mean=avg, sample_sd=sd,
                             cv_percent=100 * sd / avg if sd is not None and avg else None,
                             minimum=min(values), maximum=max(values)))
            for r in runs:
                points.append(dict(metric=metric, group=label, run=r['run'], seed=r['seed'],
                                   max_length=r['max_length'], value=r['metrics'][metric]))
    return groups, rows, points


def draw(ax, metric, groups, rows, annotate=True):
    """柱高为均值，误差线为 ±1 SD，散点为完整测试集的单次运行分数。"""
    selected = [r for r in rows if r['metric'] == metric]
    means = [r['mean'] for r in selected]
    ax.bar(range(5), means, color=COLORS, width=.62, alpha=.8)
    ax.errorbar(range(1, 5), means[1:], yerr=[r['sample_sd'] for r in selected[1:]],
                fmt='none', ecolor='#172b4d', capsize=6, elinewidth=1.7, zorder=4)
    highest = max(max(r['maximum'], r['mean'] + (r['sample_sd'] or 0)) for r in selected)
    for i, ((_, runs), summary) in enumerate(zip(groups, selected)):
        runs = sorted(runs, key=lambda r: (r['seed'] or 0, r['max_length'] or 0))
        offsets = np.linspace(-.20, .20, len(runs)) if len(runs) > 1 else [0]
        for offset, r in zip(offsets, runs):
            hollow = r['seed'] in [101, 110] and r['max_length'] == 1024
            ax.scatter(i + offset, r['metrics'][metric], marker=MARKERS[r['seed']], s=40,
                       facecolors='white' if hollow else '#172b4d', edgecolors='#172b4d', zorder=5)
        if annotate:
            top = max(summary['maximum'], summary['mean'] + (summary['sample_sd'] or 0))
            suffix = f"± {summary['sample_sd']:.4f}" if summary['sample_sd'] is not None else 'SD=N/A'
            ax.text(i, top + highest * .04, f"{summary['mean']:.4f}\n{suffix}",
                    ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(5), ['Base\nn=1', 'LoRA 3\nn=3', 'LoRA 5\nn=5', 'FT 3\nn=3', 'FT 5\nn=5'])
    ax.set_ylim(0, highest * (1.30 if annotate else 1.16))
    ax.set_ylabel('得分（原始尺度）')
    ax.set_title(metric, fontweight='bold')
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)


def save(fig, name):
    """同时保存可预览 PNG 与可编辑 SVG。"""
    for suffix in ['png', 'svg']:
        fig.savefig(OUT / 'figures' / f'{name}.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    """核对原始文件后生成统计附件，不运行训练或重新评分。"""
    records = json.loads((OUT / 'audit_results.json').read_text(encoding='utf8'))
    for r in records:
        folder = ROOT / 'outputs/benchmarks' / r['run']
        for filename, expected in r['hashes'].items():
            assert hashlib.sha256((folder / filename).read_bytes()).hexdigest() == expected
        metrics = json.loads((folder / 'metrics.json').read_text(encoding='utf8'))
        assert metrics['metrics'] == r['metrics']
    groups, rows, points = summarize(records)
    for filename, data in [('variability_summary.csv', rows), ('variability_runs.csv', points)]:
        with (OUT / filename).open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family': font.get_name(), 'axes.unicode_minus': False,
                         'axes.spines.top': False, 'axes.spines.right': False})
    legend = [Line2D([], [], color='#172b4d', marker=m, ls='', label=f'seed {s}')
              for s, m in [(42, 'o'), (43, 's'), (44, '^'), ('110 / 101', 'D')]]
    note = ('柱：均值；误差线：±1 样本标准差（ddof=1）；散点：每次运行在 1,847 条测试上的汇总分数。\n'
            '3 次组：seed 42/43/44；5 次组包含同 seed 的长度对照，只有 4 个不同 seed；空心菱形：长度 1024。\n'
            'Base 仅一次，SD 无法估计。误差线不是置信区间，也不是样本分数的最小—最大范围。')
    (OUT / 'figures').mkdir(exist_ok=True)
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(10.5, 6.8))
        draw(ax, metric, groups, rows)
        fig.suptitle(f'GPT-2 Medium · {metric} · 均值与运行间波动', fontsize=16, y=.97)
        fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, .925), ncol=4, frameon=False)
        fig.text(.06, .025, note, fontsize=9, va='bottom')
        fig.subplots_adjust(top=.80, bottom=.24, left=.10, right=.98)
        save(fig, f'variability_{metric}')
    fig, axes = plt.subplots(2, 4, figsize=(19, 10))
    for metric, ax in zip(METRICS, axes.flat):
        draw(ax, metric, groups, rows, annotate=False)
    fig.suptitle('GPT-2 Medium · 8 项指标的均值、±1 SD 与逐运行结果', fontsize=19, y=.98)
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, .95), ncol=4, frameon=False)
    fig.text(.04, .02, note, fontsize=11, va='bottom')
    fig.subplots_adjust(top=.86, bottom=.20, hspace=.38, wspace=.30)
    save(fig, 'variability_overview')
    print(json.dumps([r for r in rows if r['metric'] == 'CIDEr'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
