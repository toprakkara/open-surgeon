"""Pool kept teacher traces for v0.53 training, as training/surgical-wide-v053/plan.json says, and balance yes and no.

  python -m scripts.combine_traces_v053 --out <jsonl> <kept-unbalanced.jsonl> [...]      (needs lxml)

Every 'yes' trace is kept. A 'no' trace is kept only if its redline meets the v0.53 rule against the attorney markup
of the same clause, so no training example carries a label that v0.53 calls unreliable.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from scripts.agent_redlines_v052 import clause_pairs
from scripts.attorney_choice_v046 import endpoint, hash_key
from scripts.extract_redlinebench_paragraphs import ROOT
from scripts.surgical_verifier_v051 import parse_verdict
from scripts.validate_contract_challenge import read, sha
from scripts.wide_redlines_v053 import OUT as DATA, RETYPED_MIN, RETYPED_SHARE, kept


def redline(row):
    return [{'kind': op['kind'], 'text': op['text']} for op in json.loads(row['messages'][1]['content'])['redline']]


def clearly_wider(ops, pairs):
    """None when the clause is not in the v0.53 clause pool."""
    key = hashlib.sha256((endpoint(ops, False) + '\x00' + endpoint(ops, True)).encode()).hexdigest()
    if key not in pairs:
        return None
    base = min(kept(markup) for markup in pairs[key]['attorney'])
    retyped = base - kept(ops)
    return retyped >= RETYPED_MIN and retyped >= RETYPED_SHARE * base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('paths', type=Path, nargs='+')
    args = parser.parse_args()
    pairs, _ = clause_pairs()
    test = read(DATA / 'development/inputs.jsonl')
    test_text = {row['messages'][1]['content'] for row in test}
    test_originals = {endpoint(redline(row), False) for row in test}
    pools, seen, sources, dropped = {'yes': [], 'no': []}, {}, Counter(), Counter()
    for path in args.paths:
        for row in read(path):
            key = row['messages'][1]['content']
            verdict = parse_verdict(row['messages'][2]['content'])
            assert verdict in pools
            if key in seen:
                assert seen[key] == verdict, 'The same redline has two labels'
                dropped[f'{path.name}|{verdict}|repeat'] += 1
                continue
            seen[key] = verdict
            ops = redline(row)
            if key in test_text or endpoint(ops, False) in test_originals:
                dropped[f'{path.name}|{verdict}|clause_in_test'] += 1
                continue
            if verdict == 'no':
                wide = clearly_wider(ops, pairs)
                if not wide:
                    dropped[f"{path.name}|no|{'clause_not_in_pool' if wide is None else 'not_clearly_wider'}"] += 1
                    continue
            pools[verdict].append(row)
            sources[f'{path.name}|{verdict}'] += 1
    for label in pools:
        pools[label].sort(key=lambda row: hash_key('combine-v053/', row['messages'][1]['content']))
    size = min(len(pools['yes']), len(pools['no']))
    rows = sorted(pools['yes'][:size] + pools['no'][:size], key=lambda row: hash_key('train-order-v053/', row['messages'][1]['content']))
    used = Counter()
    chosen = {row['messages'][1]['content'] for row in rows}
    for path in args.paths:
        for row in read(path):
            if row['messages'][1]['content'] in chosen:
                chosen.discard(row['messages'][1]['content'])
                used[f"{path.name}|{parse_verdict(row['messages'][2]['content'])}"] += 1
    args.out.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    summary = {'before_balance': dict(sorted(sources.items())), 'dropped': dict(sorted(dropped.items())), 'kept_each_class': size, 'total': len(rows),
               'after_balance': dict(sorted(used.items())), 'max_user_chars': max(len(row['messages'][1]['content']) for row in rows),
               'sources': {str(p): sha(p) for p in args.paths}, 'out_sha256': sha(args.out)}
    args.out.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
