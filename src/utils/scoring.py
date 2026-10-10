"""Small-compatible scoring, retaining corpus scores and every supported row score."""
import csv
import json
import traceback
from pathlib import Path
from config import METRICS, OUTPUT_ROOT
from utils.runs import write_json

def read_predictions(path):
    records=[]
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise ValueError('No predictions to score')
    return records

def wordnet():
    import nltk
    from nltk.corpus import wordnet
    folder=OUTPUT_ROOT/'resources/nltk_data'
    folder.mkdir(parents=True,exist_ok=True)
    nltk.data.path.insert(0,str(folder))
    try:
        wordnet.ensure_loaded()
    except LookupError:
        nltk.download('wordnet',download_dir=str(folder),raise_on_error=True)
        wordnet.ensure_loaded()

def score(path, selected, output):
    rows=read_predictions(path)
    if not selected or set(selected)-set(METRICS):
        raise ValueError('Select one or more supported metrics')
    if any(row.get('error') for row in rows):
        raise ValueError('Predictions contain generation errors; inspect records before rescoring')
    gts={i:[row['target']] for i,row in enumerate(rows)}
    res={i:[row['prediction']] for i,row in enumerate(rows)}
    summary,errors,scopes={},{},{}
    per_row=[dict(id=row['id'],scores={}) for row in rows]
    for metric in selected:
        try:
            if metric=='BLEU':
                from pycocoevalcap.bleu.bleu import Bleu
                corpus,individual=Bleu(4).compute_score(gts,res)
                values={f'BLEU-{k+1}':float(v) for k,v in enumerate(corpus)}
                items=[{f'BLEU-{k+1}':float(individual[k][i]) for k in range(4)} for i in range(len(rows))]
                scope='corpus BLEU; sentence BLEU separately retained, not averaged into corpus score'
            elif metric in ('CIDEr','ROUGE-L'):
                if metric=='CIDEr':
                    from pycocoevalcap.cider.cider import Cider
                    scorer=Cider()
                else:
                    from pycocoevalcap.rouge.rouge import Rouge
                    scorer=Rouge()
                corpus,individual=scorer.compute_score(gts,res)
                values={metric:float(corpus)}
                items=[{metric:float(v)} for v in individual]
                scope='sample average; CIDEr uses this whole corpus document frequencies'
            elif metric=='NIST':
                from nltk.translate.nist_score import corpus_nist
                values={metric:float(corpus_nist([[r['target'].split()] for r in rows],
                    [r['prediction'].split() for r in rows],n=5))}
                items=[{metric:None} for _ in rows]
                scope='corpus NIST-5 only; per-sample score unavailable in this scoring profile'
            else:
                wordnet()
                from nltk.translate.meteor_score import meteor_score
                individual=[float(meteor_score([r['target'].split()],r['prediction'].split())) for r in rows]
                values={metric:sum(individual)/len(individual)}
                items=[{metric:v} for v in individual]
                scope='mean sample METEOR, NLTK tokenized whitespace with WordNet'
            summary.update(values)
            scopes[metric]=scope
            for target,item in zip(per_row,items):
                target['scores'].update(item)
            write_json(output/f'{metric}_scores.json',dict(summary=values,scope=scope,per_sample=items))
        except Exception as exc:
            errors[metric]=dict(error=str(exc),traceback=traceback.format_exc())
            write_json(output/f'{metric}_error.json',errors[metric])
        write_json(output/'metrics.json',dict(metrics=summary,errors=errors,scope=scopes,
            samples=len(rows),selected=selected,profile='small-compatible single-target'))
        with (output/'sample_scores.jsonl').open('w',encoding='utf-8') as stream:
            for row,result in zip(rows,per_row):
                stream.write(json.dumps({**row,**result},ensure_ascii=False)+'\n')
    with (output/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['metric','score'])
        writer.writeheader()
        writer.writerows(dict(metric=k,score=v) for k,v in summary.items())
    if errors:
        raise RuntimeError(f'Some metrics failed; successful scores and full predictions preserved: {list(errors)}')
    return summary
