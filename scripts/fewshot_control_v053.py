"""Control for v0.53: the same test items, with worked examples from the training split added to the instruction.

  python3 -m scripts.fewshot_control_v053 build [--dry-run]
  python3 -m scripts.fewshot_control_v053 score --run <dir> --name "<ad>"
  python3 -m scripts.fewshot_control_v053 compare <dir-a> <dir-b>

Item ids are the v0.53 test ids, so a control run can be paired with a main run. The rules are in
training/surgical-wide-v053/plan.json (fewshot_control).
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import scripts.score_surgical_agent_v052 as previous
from scripts.attorney_choice_v046 import hash_key
from scripts.surgical_verifier_v051 import SYSTEM, parse_verdict
from scripts.validate_contract_challenge import read, sha

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / 'datasets/attorney-surgical-wide-v0.53'
OUT = ROOT / 'datasets/attorney-surgical-wide-fewshot-control-v0.53'
TRACES = ROOT / 'training/surgical-wide-v053/teacher/combined-traces.jsonl'
AGENT_TRACES = ROOT / 'training/surgical-wide-v053/teacher/v053-kept-unbalanced.jsonl'
SALT = 'fewshot-control-v053/'
EACH = 4
HEADER = ('Worked examples follow. Each shows a redline from another contract and the answer a careful reviewer gave. '
          'They are examples only; judge the redline you are given on its own.')
FOOTER = 'End of examples. Now answer for the redline you are given, in the same form.'
CODE = ['scripts/fewshot_control_v053.py', 'scripts/surgical_verifier_v051.py', 'scripts/score_surgical_agent_v052.py',
        'scripts/score_surgical_verifier_v051.py']
README = '''# v0.53 control: instruction with worked examples

The questions are those of the `attorney-surgical-wide-v0.53` test (same ids, same labels). The only difference: eight
worked examples from the training split are added to the instruction (four "yes" and four "no", with the teacher's
reasoning). The aim is to see how much the v0.53 result changes when an off-the-shelf model is shown examples. The main
result does not change with this control.
'''


def pick():
    """Four yes traces from the training file and four no traces that are agent redlines of the v0.53 training split."""
    content = lambda row: row['messages'][1]['content']
    verdict = lambda row: parse_verdict(row['messages'][2]['content'])
    used = {content(row): row for row in read(TRACES)}
    yes = sorted((row for row in used.values() if verdict(row) == 'yes'), key=lambda row: hash_key(SALT + 'yes/', content(row)))[:EACH]
    agent = [used[content(row)] for row in read(AGENT_TRACES) if verdict(row) == 'no' and content(row) in used]
    no = sorted(agent, key=lambda row: hash_key(SALT + 'no/', content(row)))[:EACH]
    return [row for pair in zip(yes, no) for row in pair]


def system_prompt(examples):
    blocks = [f"Example {index}\nInput:\n{row['messages'][1]['content']}\nAnswer:\n{row['messages'][2]['content']}"
              for index, row in enumerate(examples, 1)]
    return '\n\n'.join([SYSTEM, HEADER] + blocks + [FOOTER])


def build(dry_run):
    assert not (OUT / 'protocol.json').exists(), 'Already frozen'
    examples = pick()
    test = read(MAIN / 'development/inputs.jsonl')
    assert not {row['messages'][1]['content'] for row in examples} & {row['messages'][1]['content'] for row in test}
    system = system_prompt(examples)
    inputs = [{'id': row['id'], 'messages': [{'role': 'system', 'content': system}, row['messages'][1]]} for row in test]
    report = {'control_of': 'attorney-surgical-wide-v0.53', 'n_inputs': len(inputs), 'examples': len(examples),
              'example_verdicts': [parse_verdict(row['messages'][2]['content']) for row in examples],
              'example_user_sha256': [hash_key('', row['messages'][1]['content']) for row in examples],
              'system_chars': len(system), 'traces_sha256': sha(TRACES), 'agent_traces_sha256': sha(AGENT_TRACES)}
    print(json.dumps(report, indent=1))
    if dry_run:
        print('Dry run: nothing written.')
        return
    for folder in ('development', 'test'):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    (OUT / 'development/inputs.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in inputs))
    (OUT / 'test/annotations.jsonl').write_text((MAIN / 'test/annotations.jsonl').read_text())
    (OUT / 'system-prompt.txt').write_text(system)
    (OUT / 'extraction.json').write_text(json.dumps(report, indent=2) + '\n')
    (OUT / 'README.md').write_text(README)
    names = ['README.md', 'system-prompt.txt', 'extraction.json', 'development/inputs.jsonl', 'test/annotations.jsonl']
    frozen = {name: sha(OUT / name) for name in names}
    frozen.update({'../../' + name: sha(ROOT / name) for name in CODE})
    frozen['../attorney-surgical-wide-v0.53/protocol.json'] = sha(MAIN / 'protocol.json')
    (OUT / 'protocol.json').write_text(json.dumps({'version': 'attorney-surgical-wide-fewshot-control-v0.53', 'frozen_at': datetime.now(timezone.utc).isoformat(),
                                                   'n_inputs': len(inputs), 'frozen_sha256': frozen, 'model_calls_at_freeze': 0,
                                                   'difficulty_gate_round': False}, ensure_ascii=False, indent=2) + '\n')
    print('Donduruldu:', (OUT / 'protocol.json').relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('build').add_argument('--dry-run', action='store_true')
    score = sub.add_parser('score')
    score.add_argument('--run', type=Path, required=True)
    score.add_argument('--name', required=True)
    sub.add_parser('compare').add_argument('runs', type=Path, nargs=2)
    args = parser.parse_args()
    if args.command == 'build':
        build(args.dry_run)
        return
    previous.DATA = OUT
    previous.use_dataset()
    if args.command == 'compare':
        print(json.dumps(previous.base.compare(*args.runs), indent=1))
        return
    result = previous.base.score(args.run, args.name)
    result.pop('cases')
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
