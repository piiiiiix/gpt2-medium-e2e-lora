"""Measure untruncated GPT-2 input lengths using the project's exact prompt recipe."""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'outputs/smoke/report_audit'
sys.path.insert(0, str(WORK/'deps'))
from tokenizers import Tokenizer

snapshot = Path(sys.argv[1]) if len(sys.argv)>1 else next(
    (Path.home()/'.cache/huggingface/hub/models--openai-community--gpt2-medium/snapshots').glob('*/tokenizer.json'))
tok=Tokenizer.from_file(str(snapshot))
src=WORK/'sources'
with (src/'gem_train.csv').open(encoding='utf8',newline='') as f:
    train=[{'meaning_representation':r['mr'],'target':r['ref']} for r in csv.DictReader(f)]
val=json.loads((src/'gem_validation.json').read_text(encoding='utf8'))
if isinstance(val,dict):val=list(val.values())[0]
validation=[{'meaning_representation':r['meaning_representation'],'target':r['references'][0]} for r in val if r]
result={'tokenizer_sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        'tokenizer_snapshot':snapshot.parent.name,
        'recipe':'len(encode(MR: {meaning_representation}\\nText:)) + len(encode(space + target)) + 1 EOS'}
for name,rows in [('train',train),('validation',validation)]:
    prompts=['MR: '+r['meaning_representation']+'\nText:' for r in rows]
    targets=[' '+r['target'] for r in rows]
    lens=[len(p.ids)+len(t.ids)+1 for p,t in zip(tok.encode_batch(prompts,add_special_tokens=False),
                                              tok.encode_batch(targets,add_special_tokens=False))]
    result[name]={'samples':len(lens),'max_tokens':max(lens),'over_512':sum(n>512 for n in lens),
                  'over_1024':sum(n>1024 for n in lens)}
pairs={}
records=json.loads((ROOT/'medium/audit_results.json').read_text(encoding='utf8'))
for kind,seed in [('lora',110),('fft',101)]:
    runs=[r for r in records if r['kind']==kind and r['seed']==seed]
    runs.sort(key=lambda r:r['max_length'])
    predictions=[]
    for run in runs:
        rows=[json.loads(s) for s in (ROOT/'outputs/benchmarks'/run['run']/'predictions.jsonl').read_text(encoding='utf8').splitlines()]
        predictions.append([r['prediction'] for r in rows])
    equal=sum(a==b for a,b in zip(*predictions))
    pairs[kind]={'seed':seed,'equal_predictions':equal,'different_predictions':len(predictions[0])-equal,
                 'total':len(predictions[0])}
result['length_pairs']=pairs
(ROOT/'medium/length_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
(ROOT/'medium/sources_manifest.json').write_text((src/'manifest.json').read_text(encoding='utf8'),encoding='utf8')
print(json.dumps(result,ensure_ascii=False,indent=2))
