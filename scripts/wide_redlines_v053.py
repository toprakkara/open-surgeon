"""Crosby-style test without attorneys, second build: 'no' is an agent redline clearly wider than the attorney's.

  python -m scripts.wide_redlines_v053 batches --pilot     two blind pilot batches, training clauses only
  python -m scripts.wide_redlines_v053 batches             blind batches for every other clause
  python -m scripts.wide_redlines_v053 build [--dry-run]   parse, validate, label and freeze the dataset

Needs lxml. The rules are in training/surgical-wide-v053/plan.json.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from scripts.agent_redlines_v052 import TEST_FOLD, clause_pairs, core
from scripts.attorney_choice_v046 import canonical_ops, endpoint, hash_key
from scripts.extract_redlinebench_paragraphs import ROOT
from scripts.surgical_verifier_v051 import SYSTEM, build_item
from scripts.validate_contract_challenge import read, sha

OUT = ROOT / 'datasets/attorney-surgical-wide-v0.53'
WORK = ROOT / 'training/surgical-wide-v053/agent'
SALT = 'surgical-wide-v053/'
PILOT_CLAUSES, BATCH = 64, 64
# Clearly wider: the agent strikes and retypes at least this many characters that the attorney left untouched,
# and at least this share of all the text the attorney left untouched.
RETYPED_MIN, RETYPED_SHARE = 40, 0.25
BANDS = ((0.75, 'retyped-75'), (0.5, 'retyped-50'), (0.25, 'retyped-25'))
SECTION = re.compile(r'^### ([0-9a-f]{64})[ \t]*$', re.M)
AGENT_TASK = (
    'You are an AI agent editing a contract in a document editor that records tracked changes. Your only editing tool is '
    'replace(old, new): it finds the text old in the clause and puts new in its place, and the editor then shows old as '
    'deleted and new as inserted. For each item you get the current clause and the agreed final clause. Write the replace '
    'calls that turn the current clause into the final clause. Each old must be copied exactly from the current clause, '
    'must occur in it exactly once, and the old texts of different calls must not overlap. After all calls the clause '
    'must be exactly the final clause, including every space, punctuation mark and capital letter.'
)
EARLIER = ['datasets/attorney-surgical-verifier-v0.51', 'datasets/attorney-surgical-agent-v0.52']
CODE = ['scripts/wide_redlines_v053.py', 'scripts/agent_redlines_v052.py', 'scripts/surgical_verifier_v051.py',
        'scripts/attorney_choice_v046.py', 'scripts/attorney_choice_sft_v049.py', 'scripts/prepare_attorney_choice_v046.py']
README = '''# Crosby-style surgicality test v0.53: "no" means a needlessly wide redline

Each question holds one redline. "Yes" items are the attorney's own markup. "No" items are redlines of the same clause
written by AI agents (Claude) that edit with a find-and-replace tool; only redlines clearly wider than the attorney's
are kept: the agent struck and retyped at least 40 characters, and at least a quarter, of the text the attorney left
untouched. Redlines between the two are left out of the test. The agents never saw the attorney's markup.

In Crosby's test attorneys decide what is "no"; here a rule written in advance decides. The numbers are not comparable
with Crosby's.

Source: [ContractBench](https://huggingface.co/datasets/micro1-inc/ContractBench), CC BY 4.0.
'''


def kept(ops):
    return sum(len(op['text']) for op in ops if op['kind'] == 'equal')


def parse_answers(path):
    """Each section is '### <id>' followed by one JSON object per line: {"old": ..., "new": ...}. Unreadable sections give None."""
    text = Path(path).read_text()
    marks = list(SECTION.finditer(text))
    answers = {}
    for mark, following in zip(marks, marks[1:] + [None]):
        body = text[mark.end():following.start() if following else len(text)]
        assert mark.group(1) not in answers, f'Repeated id in {path}'
        try:
            calls = [json.loads(line) for line in body.splitlines() if line.strip()]
            assert calls and all(isinstance(c, dict) and isinstance(c.get('old'), str) and isinstance(c.get('new'), str) for c in calls)
        except (ValueError, AssertionError):
            calls = None
        answers[mark.group(1)] = calls
    return answers


def apply_calls(original, calls):
    """All calls are located in the original clause and applied at once. Returns (ops, None) or (None, reason)."""
    spans = []
    for call in calls:
        if not call['old'] or original.count(call['old']) != 1:
            return None, 'old_not_unique'
        start = original.index(call['old'])
        spans.append((start, start + len(call['old']), call['new']))
    spans.sort()
    if any(first[1] > second[0] for first, second in zip(spans, spans[1:])):
        return None, 'overlap'
    ops, position = [], 0
    for start, end, new in spans:
        ops += [{'kind': 'equal', 'text': original[position:start]}, {'kind': 'delete', 'text': original[start:end]},
                {'kind': 'insert', 'text': new}]
        position = end
    ops.append({'kind': 'equal', 'text': original[position:]})
    return [op for op in ops if op['text']], None


def judge(pair, calls):
    """Returns (status, ops, retyped, share). Status: unreadable, an invalidity reason, same_as_attorney, clearly_wider or unclear."""
    if calls is None:
        return 'unreadable', None, None, None
    original, revised = core(pair)
    ops, reason = apply_calls(original, calls)
    if reason:
        return reason, None, None, None
    if endpoint(ops, True) != revised:
        return 'wrong_final_text', None, None, None
    lead, trail = pair['edges']
    ops = canonical_ops(([{'kind': 'equal', 'text': lead}] if lead else []) + ops + ([{'kind': 'equal', 'text': trail}] if trail else []))
    assert endpoint(ops, False) == pair['original'] and endpoint(ops, True) == pair['revision']
    # Measured against the attorney markup that leaves the least text untouched, so the agent is wider than every one.
    base = min(kept(markup) for markup in pair['attorney'])
    retyped = base - kept(ops)
    share = retyped / base if base else 0.0
    if any(ops == markup for markup in pair['attorney']):
        return 'same_as_attorney', ops, retyped, share
    return ('clearly_wider' if retyped >= RETYPED_MIN and share >= RETYPED_SHARE else 'unclear'), ops, retyped, share


def pilot_ids(pairs):
    train = sorted((p for p in pairs.values() if TEST_FOLD not in p['folds']), key=lambda p: hash_key(SALT + 'pilot/', p['id']))
    return [p['id'] for p in train[:PILOT_CLAUSES]]


def batches(pilot):
    pairs, _ = clause_pairs()
    chosen = pilot_ids(pairs)
    if pilot:
        ordered, name, size = [pairs[key] for key in chosen], 'pilot', PILOT_CLAUSES // 2
    else:
        rest = [p for p in pairs.values() if p['id'] not in set(chosen)]
        ordered, name, size = sorted(rest, key=lambda p: hash_key(SALT + 'batch/', p['id'])), 'batch', BATCH
    folder = WORK / 'batches'
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(0, len(ordered), size):
        with (folder / f'{name}-{index // size:03d}.jsonl').open('w') as handle:
            for pair in ordered[index:index + size]:
                current, final = core(pair)
                handle.write(json.dumps({'id': pair['id'], 'current': current, 'final': final}, ensure_ascii=False) + '\n')
    print(json.dumps({'clauses': len(ordered), 'files': -(-len(ordered) // size), 'kind': name}))


def band(share):
    return next(name for floor, name in BANDS if share >= floor)


def answer_files():
    return sorted((WORK / 'answers').glob('*/*.txt'))


def build(dry_run):
    assert not (OUT / 'protocol.json').exists(), 'Already frozen'
    pairs, sources = clause_pairs()
    answers = {}
    for path in answer_files():
        agent = path.parent.name
        for key, calls in parse_answers(path).items():
            assert (agent, key) not in answers, f'Two {agent} answers for {key}'
            answers[agent, key] = calls
    assert {key for _, key in answers} <= set(pairs), 'Answer for an unknown clause'
    agents = sorted({agent for agent, _ in answers})
    stats, items, meta = Counter(), {'test': {}, 'train': {}}, {'test': {}, 'train': {}}

    def add(part, ops, row):
        item = build_item(ops)
        known = meta[part].get(item['id'])
        assert known is None or known['label'] == row['label'], 'Same redline labelled both ways'
        if known and row['label'] == 'no':
            known['agents'] = sorted(set(known['agents']) | set(row['agents']))
            return
        items[part][item['id']], meta[part][item['id']] = item, dict(row, id=item['id'])

    for pair in pairs.values():
        part = 'test' if TEST_FOLD in pair['folds'] else 'train'
        common = {'group': hash_key(SALT + 'group/', pair['original'])[:16], 'scenario': sorted(pair['scenarios'])[0], 'clause_pair': pair['id']}
        wide = False
        for agent in agents:
            if (agent, pair['id']) not in answers:
                stats[f'{part}|{agent}|missing'] += 1
                continue
            status, ops, retyped, share = judge(pair, answers[agent, pair['id']])
            stats[f'{part}|{agent}|{status}'] += 1
            if status == 'clearly_wider':
                wide = True
                add(part, ops, dict(common, label='no', cell=band(share), retyped=retyped, share=round(share, 3), agents=[agent]))
        if wide:
            stats[f'{part}|clauses_with_no_item'] += 1
            for markup in pair['attorney']:
                add(part, markup, dict(common, label='yes', cell='attorney'))
    order = lambda part: sorted(items[part], key=lambda key: hash_key(SALT + 'order/', key))
    rows = {part: ([items[part][k] for k in order(part)], [meta[part][k] for k in order(part)]) for part in items}
    assert not {m['group'] for m in rows['test'][1]} & {m['group'] for m in rows['train'][1]}
    text = lambda rows_: {r['messages'][1]['content'] for r in rows_}
    test_text, earlier_train, earlier_test = text(rows['test'][0]), set(), set()
    for data in EARLIER:
        earlier_train |= text(read(ROOT / data / 'train/prompts.jsonl'))
        earlier_test |= text(read(ROOT / data / 'development/inputs.jsonl'))
    assert not test_text & (earlier_train | text(rows['train'][0])), 'A test redline is in training data'
    labels = lambda part: dict(sorted(Counter(f"{m['label']}|{m['cell']}" for m in rows[part][1]).items()))
    report = {'status': 'frozen_before_any_verifier_output_on_these_redlines',
              'rule': {'retyped_min_chars': RETYPED_MIN, 'retyped_min_share_of_attorney_kept_text': RETYPED_SHARE},
              'agents': agents, 'agent_counts': dict(sorted(stats.items())), 'test_items': len(rows['test'][0]), 'test_labels': labels('test'),
              'train_items': len(rows['train'][0]), 'train_labels': labels('train'),
              'test_inputs_also_in_earlier_tests': len(test_text & earlier_test), 'test_inputs_in_earlier_training': 0,
              'test_max_user_chars': max((len(r['messages'][1]['content']) for r in rows['test'][0]), default=0),
              'agent_answers': len(answers), 'clause_pairs': len(pairs), 'source_sha256': sources}
    print(json.dumps({k: v for k, v in report.items() if k != 'source_sha256'}, ensure_ascii=False, indent=1))
    if dry_run:
        print('Dry run: nothing written.')
        return
    for folder in ('development', 'test', 'train'):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    write = lambda path, rows_: path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows_))
    write(OUT / 'development/inputs.jsonl', rows['test'][0])
    write(OUT / 'test/annotations.jsonl', rows['test'][1])
    write(OUT / 'train/prompts.jsonl', rows['train'][0])
    write(OUT / 'train/labels.jsonl', rows['train'][1])
    (OUT / 'system-prompt.txt').write_text(SYSTEM)
    (OUT / 'agent-task.txt').write_text(AGENT_TASK)
    (OUT / 'extraction.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'README.md').write_text(README)
    names = ['README.md', 'system-prompt.txt', 'agent-task.txt', 'extraction.json', 'development/inputs.jsonl',
             'test/annotations.jsonl', 'train/prompts.jsonl', 'train/labels.jsonl']
    frozen = {name: sha(OUT / name) for name in names}
    frozen.update({'../../' + name: sha(ROOT / name) for name in CODE})
    frozen.update({f'../../{p.relative_to(ROOT)}': sha(p) for p in answer_files()})
    (OUT / 'protocol.json').write_text(json.dumps({'version': 'attorney-surgical-wide-v0.53', 'frozen_at': datetime.now(timezone.utc).isoformat(),
                                                   'n_inputs': len(rows['test'][0]), 'frozen_sha256': frozen, 'model_calls_at_freeze': 0,
                                                   'difficulty_gate_round': False}, ensure_ascii=False, indent=2) + '\n')
    print('Donduruldu:', (OUT / 'protocol.json').relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('batches').add_argument('--pilot', action='store_true')
    sub.add_parser('build').add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    batches(args.pilot) if args.command == 'batches' else build(args.dry_run)


if __name__ == '__main__':
    main()
