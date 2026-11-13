"""Pool kept teacher traces from several datasets and balance yes and no, as training/surgical-agent-v052/plan.json says.

  python3 -m scripts.combine_traces_v052 --out <jsonl> <kept-unbalanced.jsonl> [...]
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.attorney_choice_v046 import hash_key
from scripts.surgical_verifier_v051 import parse_verdict
from scripts.validate_contract_challenge import read, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('paths', type=Path, nargs='+')
    args = parser.parse_args()
    # An agent redline that matches the attorney is the same text as a v0.51 item. The first file's trace is kept
    # and the repeat is counted, so no redline is trained on twice.
    pools, seen, sources, repeats = {'yes': [], 'no': []}, {}, Counter(), Counter()
    for path in args.paths:
        for row in read(path):
            key = row['messages'][1]['content']
            verdict = parse_verdict(row['messages'][2]['content'])
            assert verdict in pools
            if key in seen:
                assert seen[key] == verdict, 'The same redline has two labels'
                repeats[f'{path.name}|{verdict}'] += 1
                continue
            seen[key] = verdict
            pools[verdict].append(row)
            sources[f'{path.name}|{verdict}'] += 1
    for label in pools:
        pools[label].sort(key=lambda row: hash_key('combine-v052/', row['messages'][1]['content']))
    size = min(len(pools['yes']), len(pools['no']))
    rows = sorted(pools['yes'][:size] + pools['no'][:size], key=lambda row: hash_key('train-order-v052/', row['messages'][1]['content']))
    args.out.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    summary = {'before_balance': dict(sources), 'repeats_skipped': dict(repeats), 'kept_each_class': size, 'total': len(rows),
               'sources': {str(p): sha(p) for p in args.paths}, 'out_sha256': sha(args.out)}
    args.out.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
