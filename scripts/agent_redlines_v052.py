"""Crosby-style data without attorneys: an AI agent proposes the redline, attorney match decides the label.

  python3 -m scripts.agent_redlines_v052 batches --size 64      blind agent batches (original + revised only)
  python3 -m scripts.agent_redlines_v052 build [--dry-run]       parse, validate, label and freeze the dataset
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from scripts.attorney_choice_sft_v049 import assign_folds
from scripts.attorney_choice_v046 import canonical_ops, endpoint, hash_key
from scripts.extract_redlinebench_paragraphs import ROOT
from scripts.prepare_attorney_choice_v046 import build_population
from scripts.surgical_verifier_v051 import SYSTEM, build_item
from scripts.validate_contract_challenge import sha

OUT = ROOT / 'datasets/attorney-surgical-agent-v0.52'
WORK = ROOT / 'training/surgical-agent-v052/agent'
TEST_FOLD = 2
SALT = 'surgical-agent-v052/'
SECTION = re.compile(r'^### ([0-9a-f]{64})[ \t]*$', re.M)
MARK = re.compile(r'\[-(.*?)-\]|\{\+(.*?)\+\}', re.S)
AGENT_TASK = (
    'You are a contract-review assistant. For each item you get the original clause and the agreed revised clause. '
    'Prepare the tracked-change redline you would send to the other side, showing how the original becomes the revised '
    'clause. Write the clause once, copying unchanged text exactly, and wrap deleted text in [- -] and inserted text in '
    '{+ +}. Removing every deletion and keeping every insertion must give exactly the revised clause; keeping every '
    'deletion and removing every insertion must give exactly the original. Keep every space, punctuation mark and capital '
    'letter exactly as in the texts.'
)
CODE = ['scripts/agent_redlines_v052.py', 'scripts/surgical_verifier_v051.py', 'scripts/attorney_choice_v046.py',
        'scripts/attorney_choice_sft_v049.py', 'scripts/prepare_attorney_choice_v046.py']


def clause_pairs():
    """Unique (original, revised) pairs with every attorney markup of that pair, and the shared edge whitespace."""
    limits = json.loads((ROOT / 'datasets/attorney-markup-choice-v0.46/selection-plan.json').read_text())['limits']
    split = json.loads((ROOT / 'datasets/attorney-choice-sft-v0.49/selection-plan.json').read_text())['split']
    rows, _, sources = build_population(limits)
    folds = assign_folds(sorted({r['scenario'] for r in rows}), split['folds'], split['salt'])
    pairs = {}
    for row in rows:
        key = hashlib.sha256((row['original'] + '\x00' + row['revision']).encode()).hexdigest()
        pair = pairs.setdefault(key, {'id': key, 'original': row['original'], 'revision': row['revision'], 'attorney': [],
                                      'scenarios': set(), 'folds': set()})
        pair['attorney'].append(row['attorney'])
        pair['scenarios'].add(row['scenario'])
        pair['folds'].add(folds[row['scenario']])
    # A clause seen in more than one fold, or a training clause whose original text is in the test fold, is left out.
    test_originals = {p['original'] for p in pairs.values() if TEST_FOLD in p['folds']}
    dropped = Counter()
    for key in list(pairs):
        pair = pairs[key]
        if len(pair['folds']) > 1:
            dropped['spans_folds'] += 1
            del pairs[key]
        elif TEST_FOLD not in pair['folds'] and pair['original'] in test_originals:
            dropped['train_original_in_test'] += 1
            del pairs[key]
    sources = dict(sources, dropped_clause_pairs=dict(dropped))
    for pair in pairs.values():
        original, revision = pair['original'], pair['revision']
        lead = len(original) - len(original.lstrip())
        trail = len(original) - len(original.rstrip())
        same = (original[:lead] == revision[:len(revision) - len(revision.lstrip())]
                and original[len(original) - trail:] == revision[len(revision.rstrip()):])
        pair['edges'] = (original[:lead], original[len(original) - trail:]) if same and (lead or trail) else ('', '')
    return pairs, sources


def core(pair):
    lead, trail = pair['edges']
    cut = lambda text: text[len(lead):len(text) - len(trail)]
    return cut(pair['original']), cut(pair['revision'])


def parse_inline(text):
    ops, position = [], 0
    for match in MARK.finditer(text):
        if match.start() > position:
            ops.append({'kind': 'equal', 'text': text[position:match.start()]})
        deleted, inserted = match.group(1), match.group(2)
        ops.append({'kind': 'delete', 'text': deleted} if deleted is not None else {'kind': 'insert', 'text': inserted})
        position = match.end()
    if position < len(text):
        ops.append({'kind': 'equal', 'text': text[position:]})
    return [op for op in ops if op['text']]


def parse_answers(path):
    text = Path(path).read_text()
    marks = list(SECTION.finditer(text))
    answers = {}
    for mark, following in zip(marks, marks[1:] + [None]):
        body = text[mark.end():following.start() if following else len(text)]
        body = body[1:] if body.startswith('\n') else body
        body = body.rstrip('\n')
        assert mark.group(1) not in answers, f'Repeated id in {path}'
        answers[mark.group(1)] = body
    return answers


def batches(size):
    pairs, _ = clause_pairs()
    ordered = sorted(pairs.values(), key=lambda p: hash_key(SALT + 'batch/', p['id']))
    folder = WORK / 'batches'
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(0, len(ordered), size):
        with (folder / f'batch-{index // size:03d}.jsonl').open('w') as handle:
            for pair in ordered[index:index + size]:
                original, revised = core(pair)
                handle.write(json.dumps({'id': pair['id'], 'original': original, 'revised': revised}, ensure_ascii=False) + '\n')
    print(json.dumps({'pairs': len(ordered), 'batches': -(-len(ordered) // size)}))


def build(dry_run):
    assert not (OUT / 'protocol.json').exists(), 'Already frozen'
    pairs, sources = clause_pairs()
    answers = {}
    for path in sorted((WORK / 'answers').glob('batch-*.txt')):
        for key, body in parse_answers(path).items():
            assert key not in answers, f'Two agent answers for {key}'
            answers[key] = body
    stats, items, meta = Counter(), {'test': [], 'train': []}, {'test': [], 'train': []}
    for pair in sorted(pairs.values(), key=lambda p: hash_key(SALT + 'order/', p['id'])):
        part = 'test' if TEST_FOLD in pair['folds'] else 'train'
        if pair['id'] not in answers:
            stats[f'{part}|missing'] += 1
            continue
        ops = parse_inline(answers[pair['id']])
        original, revised = core(pair)
        if endpoint(ops, False) != original or endpoint(ops, True) != revised:
            stats[f'{part}|invalid'] += 1
            continue
        lead, trail = pair['edges']
        ops = ([{'kind': 'equal', 'text': lead}] if lead else []) + ops + ([{'kind': 'equal', 'text': trail}] if trail else [])
        ops = canonical_ops(ops)
        assert endpoint(ops, False) == pair['original'] and endpoint(ops, True) == pair['revision']
        label = 'yes' if any(ops == attorney for attorney in pair['attorney']) else 'no'
        stats[f'{part}|{label}'] += 1
        item = build_item(ops)
        items[part].append(item)
        meta[part].append({'id': item['id'], 'label': label, 'group': hash_key(SALT + 'group/', pair['original'])[:16],
                           'cell': 'agent', 'scenario': sorted(pair['scenarios'])[0], 'clause_pair': pair['id'],
                           'attorney_markups': len(pair['attorney'])})
    for part in items:
        assert len({i['id'] for i in items[part]}) == len(items[part])
    assert not {m['group'] for m in meta['test']} & {m['group'] for m in meta['train']}
    report = {'status': 'frozen_before_any_model_output_on_these_redlines', 'counts': dict(sorted(stats.items())),
              'test_items': len(items['test']), 'train_items': len(items['train']), 'agent_answers': len(answers),
              'clause_pairs': len(pairs), 'source_sha256': sources}
    print(json.dumps({k: v for k, v in report.items() if k != 'source_sha256'}, ensure_ascii=False, indent=1))
    if dry_run:
        print('Dry run: nothing written.')
        return
    for folder in ('development', 'test', 'train'):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    write = lambda path, rows: path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    write(OUT / 'development/inputs.jsonl', items['test'])
    write(OUT / 'test/annotations.jsonl', meta['test'])
    write(OUT / 'train/prompts.jsonl', items['train'])
    write(OUT / 'train/labels.jsonl', meta['train'])
    (OUT / 'system-prompt.txt').write_text(SYSTEM)
    (OUT / 'agent-task.txt').write_text(AGENT_TASK)
    (OUT / 'extraction.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'README.md').write_text(
        '# Crosby-style agent redline test v0.52\n\nEach question is a redline proposed by an AI agent (Claude Sonnet) '
        'that did not see the attorney\'s markup. The label is yes when the redline is exactly the attorney\'s markup and no '
        'otherwise. This is Crosby\'s training label; Crosby\'s test is a private set reviewed by attorneys. The numbers are '
        'not comparable.\n\n'
        'Source: [ContractBench](https://huggingface.co/datasets/micro1-inc/ContractBench), CC BY 4.0.\n')
    names = ['README.md', 'system-prompt.txt', 'agent-task.txt', 'extraction.json', 'development/inputs.jsonl',
             'test/annotations.jsonl', 'train/prompts.jsonl', 'train/labels.jsonl']
    frozen = {name: sha(OUT / name) for name in names}
    frozen.update({'../../' + name: sha(ROOT / name) for name in CODE})
    frozen.update({f'../../{p.relative_to(ROOT)}': sha(p) for p in sorted((WORK / 'answers').glob('batch-*.txt'))})
    (OUT / 'protocol.json').write_text(json.dumps({'version': 'attorney-surgical-agent-v0.52', 'frozen_at': datetime.now(timezone.utc).isoformat(),
                                                   'n_inputs': len(items['test']), 'frozen_sha256': frozen, 'model_calls_at_freeze': 0,
                                                   'difficulty_gate_round': False}, ensure_ascii=False, indent=2) + '\n')
    print('Donduruldu:', (OUT / 'protocol.json').relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('batches').add_argument('--size', type=int, default=64)
    sub.add_parser('build').add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    batches(args.size) if args.command == 'batches' else build(args.dry_run)


if __name__ == '__main__':
    main()
