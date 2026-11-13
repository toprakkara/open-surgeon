"""Run release/usage_example.py as published, on Modal, and compare it with the recorded v0.53 measurement.

  uvx --from modal==1.6.3 modal run scripts/modal_usage_check_v053.py --out training/surgical-wide-v053/usage-check

It calls the same load() and judge() the model card shows, on the card's own example and on a few test items
whose measured outputs are already in trained-27b/predictions.jsonl. Paid: a few minutes of one H100.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import time

import modal

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets/attorney-surgical-wide-v0.53'
RUN = ROOT / 'training/surgical-wide-v053/trained-27b'
EXAMPLE = ROOT / 'release/usage_example.py'
CACHE = '/cache'
ADAPTER = 'v053-27b'
EACH = 3
VERDICT = re.compile(r'\{\s*"surgical"\s*:\s*"(yes|no)"\s*\}\s*$')

app = modal.App('open-surgeon-usage-check-v053')
image = (modal.Image.debian_slim(python_version='3.12').pip_install('torch', 'transformers==5.22.1', 'peft', 'accelerate', 'safetensors')
         .add_local_file(str(EXAMPLE), '/root/usage_example.py'))
volume = modal.Volume.from_name('open-surgeon-v049')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def verdict(text):
    match = VERDICT.search(text.strip())
    return match.group(1) if match else None


@app.function(image=image, gpu='H100', timeout=20 * 60, volumes={CACHE: volume})
def check(system, items):
    os.environ['HF_HOME'] = f'{CACHE}/hf'
    volume.reload()
    import sys
    sys.path.insert(0, '/root')
    import peft
    import torch
    import transformers
    import usage_example
    began = time.monotonic()
    tokenizer, model = usage_example.load(f'{CACHE}/adapters/{ADAPTER}')
    loaded = round(time.monotonic() - began)
    outputs = {'card_example': usage_example.judge(tokenizer, model, system, usage_example.EXAMPLE)}
    for item in items:
        outputs[item['id']] = usage_example.judge(tokenizer, model, system, item['redline'])
    return {'outputs': outputs, 'load_seconds': loaded, 'total_seconds': round(time.monotonic() - began),
            'versions': {'torch': str(torch.__version__), 'transformers': str(transformers.__version__), 'peft': str(peft.__version__),
                         'device': torch.cuda.get_device_name(0)}}


@app.local_entrypoint()
def main(out: str):
    run = Path(out)
    run.mkdir(parents=True, exist_ok=False)
    labels = {row['id']: row['label'] for row in read(DATA / 'test/annotations.jsonl')}
    recorded = {row['id']: row['output'] for row in read(RUN / 'predictions.jsonl')}
    chosen, count = [], {'yes': 0, 'no': 0}
    for row in read(DATA / 'development/inputs.jsonl'):        # the first three yes and three no items in test order
        if count[labels[row['id']]] < EACH:
            count[labels[row['id']]] += 1
            chosen.append({'id': row['id'], 'redline': json.loads(row['messages'][1]['content'])['redline']})
    result = check.remote((DATA / 'system-prompt.txt').read_text(), chosen)
    rows = [{'id': item['id'], 'label': labels[item['id']], 'verdict': verdict(result['outputs'][item['id']]),
             'recorded_verdict': verdict(recorded[item['id']]), 'same_text_as_recorded': result['outputs'][item['id']] == recorded[item['id']]}
            for item in chosen]
    summary = {'usage_example_sha256': sha(EXAMPLE), 'adapter': ADAPTER, 'card_example_output': result['outputs']['card_example'],
               'card_example_verdict': verdict(result['outputs']['card_example']), 'test_items': rows,
               'same_verdict_as_measurement': sum(r['verdict'] == r['recorded_verdict'] for r in rows), 'n': len(rows),
               'note': 'The measurement decoded in batches of 8 with left padding; this run decodes one item at a time, so texts may differ slightly.',
               'load_seconds': result['load_seconds'], 'total_seconds': result['total_seconds'], 'versions': result['versions'],
               'completed_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    (run / 'result.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ('card_example_verdict', 'same_verdict_as_measurement', 'n', 'load_seconds', 'total_seconds')}))
