"""Fetch the exact audited source snapshots into the ignored report work directory."""
import hashlib
import json
from pathlib import Path
import urllib.request

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'medium/sources_manifest.json').read_text(encoding='utf8'))
destination=root/'outputs/smoke/report_audit/sources'
destination.mkdir(parents=True,exist_ok=True)
for item in manifest:
    path=destination/item['file']
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']:
        continue
    with urllib.request.urlopen(item['url'],timeout=60) as response:
        data=response.read()
    if hashlib.sha256(data).hexdigest()!=item['sha256']:
        raise ValueError(f"Upstream content changed: {item['url']}; audited snapshot required")
    path.write_bytes(data)
(destination/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
print(f'Verified {len(manifest)} source snapshots.')
