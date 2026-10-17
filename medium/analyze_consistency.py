"""Audit saved benchmarks without loading models or changing source results.

Dependencies: numpy, matplotlib, nltk, future, pycocoevalcap==1.2.
For this report, pycocoevalcap is isolated in outputs/smoke/report_audit/deps.
Upstream pymteval.py and dataset snapshots are listed in sources_manifest.json.
"""
import collections
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
WORK = ROOT / 'outputs/smoke/report_audit'
sys.path[:0] = [str(WORK / 'deps'), str(WORK / 'sources')]
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from nltk.translate.nist_score import corpus_nist
from pymteval import BLEUScore, NISTScore


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(rows, multi):
    refs = [r['references'] if multi else [r['target']] for r in rows]
    preds = [r['prediction'] for r in rows]
    gts = dict(enumerate(refs))
    res = {i: [s] for i, s in enumerate(preds)}
    with contextlib.redirect_stdout(io.StringIO()):
        bs, _ = Bleu(4).compute_score(gts, res)
        rouge, _ = Rouge().compute_score(gts, res)
        cider, _ = Cider().compute_score(gts, res)
    b, n = BLEUScore(), NISTScore()
    for p, rs in zip(preds, refs):
        b.append(p, rs)
        n.append(p, rs)
    return {'COCO_BLEU4': bs[3], 'ROUGE-L': rouge, 'CIDEr': cider,
            'NLTK_NIST': corpus_nist([[s.split() for s in rs] for rs in refs],
                                     [p.split() for p in preds], n=5),
            'MTEval_BLEU': b.score(), 'MTEval_NIST': n.score()}


records = []
first = None
for folder in sorted((ROOT / 'outputs/benchmarks').iterdir()):
    if not folder.is_dir() or not (folder / 'metrics.json').exists():
        continue
    c, m, st, ds = [read(folder / f) for f in
                    ('config.json', 'metrics.json', 'status.json', 'data_summary.json')]
    rows = [json.loads(s) for s in (folder / 'predictions.jsonl').read_text(encoding='utf8').splitlines()]
    assert st['status'] == 'completed' and not c['smoke'] and not m['errors']
    assert len(rows) == m['samples'] == ds['count'] == 1847
    assert all(not r.get('error') and isinstance(r['prediction'], str) and r['prediction'] for r in rows)
    assert all(r['target'] == r['references'][0] for r in rows)
    identity = [(r['id'], r['meaning_representation'], r['target'], r['references']) for r in rows]
    if first is None:
        first = identity
    assert identity == first
    tc = c['model'].get('training_config', {})
    kind = c['model']['type']
    label = 'Base' if kind == 'base' else f"{kind.upper()}-s{tc['seed']}-L{tc['max_length']}"
    print('Scoring', label, flush=True)
    scores = {name: score(rows, multi) for name, multi in [('single', False), ('multi', True)]}
    # Recompute original metrics to distinguish a reporting error from protocol differences.
    for ours, original in [('COCO_BLEU4', 'BLEU-4'), ('ROUGE-L', 'ROUGE-L'),
                           ('CIDEr', 'CIDEr'), ('NLTK_NIST', 'NIST')]:
        assert abs(scores['single'][ours] - m['metrics'][original]) < 1e-10, (label, ours)
    records.append(dict(label=label, run=folder.name, kind=kind, seed=tc.get('seed'),
                        max_length=tc.get('max_length'), training_config=tc,
                        generation_config={k: v for k, v in c.items() if k != 'model'},
                        dataset=ds, environment=read(folder/'environment.json'),
                        metrics=m['metrics'], rescoring=scores,
                        hashes={f: digest(folder/f) for f in ['config.json','metrics.json','predictions.jsonl']},
                        training_path=c['model'].get('training_run'),
                        prediction_text_sha256=hashlib.sha256(json.dumps([r['prediction'] for r in rows], ensure_ascii=False).encode()).hexdigest(),
                        examples=rows[:2]))
    (OUT / 'audit_results.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf8')

# Compare dataset structures to the actual snapshots used by the official pipeline.
src = WORK / 'sources'
orig = [s.split('||', 1) for s in (src/'lora_test.txt').read_text(encoding='utf8').splitlines() if s]
train = [s.split('||', 1) for s in (src/'lora_train.txt').read_text(encoding='utf8').splitlines() if s]
with (src/'gem_train.csv').open(encoding='utf8', newline='') as f:
    gem_train = list(csv.DictReader(f))
gem = read(src/'gem_test.json')
if isinstance(gem, dict):
    gem = list(gem.values())[0]
assert [(g['meaning_representation'], g['references']) for g in gem] == [(x[1], x[3]) for x in first]
summary = dict(runs=len(records), rows=len(first), unique_mr=len({x[1] for x in first}),
               reference_count_histogram=dict(sorted(collections.Counter(len(x[3]) for x in first).items())),
               total_references=sum(len(x[3]) for x in first),
               multiple_reference_rows=sum(len(x[3])>1 for x in first),
               official_test_rows=len(orig), official_test_unique_contexts=len({x[0] for x in orig}),
               official_train_rows=len(train), gem_train_rows=len(gem_train))
summary['source_code_commit'] = '117a9d1'
(OUT/'data_audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
(OUT/'sources_manifest.json').write_text((src/'manifest.json').read_text(encoding='utf8'), encoding='utf8')
with (OUT/'run_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
    fields=['label','run','kind','seed','max_length']+list(records[0]['metrics'])
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for r in records:w.writerow({**{k:r[k] for k in fields[:5]},**r['metrics']})
with (OUT/'rescoring.csv').open('w',encoding='utf-8-sig',newline='') as f:
    fields=['label','profile']+list(records[0]['rescoring']['single'])
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for r in records:
        for profile, metrics in r['rescoring'].items():w.writerow(dict(label=r['label'],profile=profile,**metrics))
print(json.dumps(summary, ensure_ascii=False),flush=True)
