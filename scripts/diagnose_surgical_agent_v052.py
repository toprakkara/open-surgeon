"""Post-hoc look at how the agent's 'no' redlines differ from the attorney's. Not part of the registered scoring.

  python -m scripts.diagnose_surgical_agent_v052 <run-dir> [...]      (needs lxml)

Each 'no' test item is put in one group:
  whitespace_only   the same non-whitespace characters are marked as in an attorney markup; only spaces moved
  agent_marks_less  the agent marks fewer changed characters than the smallest attorney markup
  agent_marks_more  the agent marks more
  same_amount       the same number of changed characters, placed differently
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from scripts.agent_redlines_v052 import clause_pairs
from scripts.surgical_verifier_v051 import parse_verdict
from scripts.validate_contract_challenge import read, sha

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets/attorney-surgical-agent-v0.52'
OUT = ROOT / 'training/surgical-agent-v052/diagnostic.json'


def marks(ops):
    """Positions of the non-whitespace characters marked deleted (in the original) and inserted (in the revision)."""
    source = target = 0
    deleted, inserted = set(), set()
    for op in ops:
        for char in op['text']:
            if op['kind'] != 'insert':
                if op['kind'] == 'delete' and not char.isspace():
                    deleted.add(source)
                source += 1
            if op['kind'] != 'delete':
                if op['kind'] == 'insert' and not char.isspace():
                    inserted.add(target)
                target += 1
    return frozenset(deleted), frozenset(inserted)


def changed(ops):
    return sum(len(op['text']) for op in ops if op['kind'] != 'equal')


def group(agent, attorney):
    if any(marks(agent) == marks(markup) for markup in attorney):
        return 'whitespace_only'
    size, smallest = changed(agent), min(changed(markup) for markup in attorney)
    return 'agent_marks_less' if size < smallest else 'agent_marks_more' if size > smallest else 'same_amount'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', type=Path, nargs='+')
    args = parser.parse_args()
    pairs, _ = clause_pairs()
    redlines = {row['id']: json.loads(row['messages'][1]['content'])['redline'] for row in read(DATA / 'development/inputs.jsonl')}
    runs = {run.name: {row['id']: parse_verdict(row['output']) for row in read(run / 'predictions.jsonl')} for run in args.runs}
    counts, caught = Counter(), defaultdict(Counter)
    for row in read(DATA / 'test/annotations.jsonl'):
        if row['label'] != 'no':
            continue
        name = group(redlines[row['id']], pairs[row['clause_pair']]['attorney'])
        counts[name] += 1
        for run, verdicts in runs.items():
            caught[name][run] += verdicts[row['id']] == 'no'
    result = {'note': 'Post-hoc diagnostic written after the Astra, heuristic and v0.51 model outputs were seen. No label or score depends on it.',
              'no_items': sum(counts.values()), 'groups': {name: {'items': counts[name], 'called_no': dict(caught[name])} for name in sorted(counts)},
              'runs': {run.name: sha(run / 'predictions.jsonl') for run in args.runs}}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result['groups'], indent=1))


if __name__ == '__main__':
    main()
