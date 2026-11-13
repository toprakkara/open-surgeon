"""Freeze the v0.51 Crosby-style test (fold 2) and training items (folds 0 and 1) before any model output on them."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json

from scripts.attorney_choice_v046 import canonical_ops, hash_key
from scripts.extract_redlinebench_paragraphs import ROOT
from scripts.prepare_attorney_choice_folds_v050 import build
from scripts.surgical_verifier_v051 import SYSTEM, build_item
from scripts.validate_contract_challenge import read, sha

OUT = ROOT / 'datasets/attorney-surgical-verifier-v0.51'
FOLD = 2
SALT = 'surgical-verifier-v051/'
ROUNDS = ['datasets/attorney-markup-choice-v0.46', 'datasets/attorney-markup-balanced-v0.47']
CODE = ['scripts/surgical_verifier_v051.py', 'scripts/prepare_surgical_verifier_v051.py', 'scripts/prepare_attorney_choice_folds_v050.py',
        'scripts/prepare_attorney_choice_sft_v049.py', 'scripts/attorney_choice_v046.py', 'scripts/attorney_choice_sft_v049.py']
README = '''# Crosby-style surgicality verifier v0.51

Each question holds one redline; the model writes a short reasoning and answers yes or no to "is this surgical?". The
label is attorney match: yes when the redline is exactly the attorney's markup, no when it is another, automatic markup of
the same change. This is Crosby's training label; Crosby's test is a private set relabelled by attorneys, so the numbers
are not comparable with Crosby's. The result measures likeness to the attorney.

Test: fold 3 (fold index 2) of v0.46 and v0.47, with the `targeted` diff as the "no" example. Training: scenarios of
folds 1 and 2, with the `word`, `character` and `whole` diffs as "no" examples. Test scenarios and paragraphs are kept
out of training.

Source: [ContractBench](https://huggingface.co/datasets/micro1-inc/ContractBench), CC BY 4.0.
'''


def items_from_pairs(pairs, group_key):
    """Each pair gives an attorney item (yes) and a distractor item (no). Conflicting or repeated content is resolved here."""
    items = {}
    for pair in pairs:
        for ops, label in ((pair['attorney'], 'yes'), (pair['distractor'], 'no')):
            item = build_item(ops)
            meta = {'id': item['id'], 'label': label, 'group': hash_key(SALT + 'group/', group_key(pair))[:16],
                    'mode': 'attorney' if label == 'yes' else pair['mode'], 'cell': pair['cell'], 'scenario': pair['scenario']}
            if item['id'] in items:
                assert items[item['id']][1]['label'] == label, 'Same redline labelled both ways'
                continue
            items[item['id']] = (item, meta)
    ordered = sorted(items.values(), key=lambda pair: hash_key(SALT + 'order/', pair[0]['id']))
    return [item for item, _ in ordered], [meta for _, meta in ordered]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    assert not (OUT / 'protocol.json').exists(), 'Already frozen'
    summary, train, dev, _, test_annotations = build(FOLD)
    test_ids = {row['id'] for row in test_annotations}
    test_pairs = []
    for data in ROUNDS:
        for row in read(ROOT / data / 'selected.jsonl'):
            if row['id'] in test_ids:
                test_pairs.append({'attorney': canonical_ops(row['attorney_operations']), 'distractor': canonical_ops(row['algorithm_operations']),
                                   'mode': 'targeted', 'cell': row['cell'], 'scenario': row['scenario'], 'original': row['original']})
    assert len(test_pairs) == len(test_ids)
    test_items, test_meta = items_from_pairs(test_pairs, lambda pair: pair['original'])
    train_items, train_meta = items_from_pairs(train + dev, lambda pair: pair['original'])
    test_text = {item['messages'][1]['content'] for item in test_items}
    assert not test_text & {item['messages'][1]['content'] for item in train_items}
    assert not {m['scenario'] for m in test_meta} & {m['scenario'] for m in train_meta}
    assert not {p['original'] for p in test_pairs} & {p['original'] for p in train + dev}
    report = {'status': 'frozen_before_any_model_output', 'test_fold': FOLD, 'test_items': len(test_items),
              'test_labels': dict(Counter(m['label'] for m in test_meta)), 'test_pairs': len(test_pairs),
              'train_items': len(train_items), 'train_labels': dict(Counter(m['label'] for m in train_meta)),
              'train_by_mode': dict(Counter(m['mode'] for m in train_meta)),
              'test_max_user_chars': max(len(i['messages'][1]['content']) for i in test_items),
              'train_max_user_chars': max(len(i['messages'][1]['content']) for i in train_items),
              'source_fold_summary_sha256': summary['source_sha256']}
    print(json.dumps({k: v for k, v in report.items() if k != 'source_fold_summary_sha256'}, ensure_ascii=False, indent=1))
    if args.dry_run:
        print('Dry run: nothing written.')
        return
    for folder in ('development', 'test', 'train'):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    write = lambda path, rows: path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    write(OUT / 'development/inputs.jsonl', test_items)
    write(OUT / 'test/annotations.jsonl', test_meta)
    write(OUT / 'train/prompts.jsonl', train_items)
    write(OUT / 'train/labels.jsonl', train_meta)
    (OUT / 'system-prompt.txt').write_text(SYSTEM)
    (OUT / 'extraction.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'README.md').write_text(README)
    names = ['README.md', 'system-prompt.txt', 'extraction.json', 'development/inputs.jsonl', 'test/annotations.jsonl',
             'train/prompts.jsonl', 'train/labels.jsonl']
    frozen = {name: sha(OUT / name) for name in names}
    frozen.update({'../../' + name: sha(ROOT / name) for name in CODE})
    (OUT / 'protocol.json').write_text(json.dumps({'version': 'attorney-surgical-verifier-v0.51', 'frozen_at': datetime.now(timezone.utc).isoformat(),
                                                   'n_inputs': len(test_items), 'frozen_sha256': frozen, 'model_calls_at_freeze': 0,
                                                   'difficulty_gate_round': False}, ensure_ascii=False, indent=2) + '\n')
    print('Donduruldu:', (OUT / 'protocol.json').relative_to(ROOT))


if __name__ == '__main__':
    main()
