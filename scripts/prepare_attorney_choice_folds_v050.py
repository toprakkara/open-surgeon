"""Build the fold 1 and fold 2 training data and test cases with the v0.49 selection plan, before any student output on them."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json

from scripts.attorney_choice_sft_v049 import assert_disjoint, assign_folds, training_example
from scripts.attorney_choice_v046 import hash_key
from scripts.extract_redlinebench_paragraphs import ROOT
from scripts.prepare_attorney_choice_sft_v049 import FROZEN_CODE, README, ROUNDS, candidates, is_dev, load_test, take
from scripts.prepare_attorney_choice_v046 import build_population
from scripts.validate_contract_challenge import read, sha

PLAN = ROOT / 'datasets/attorney-choice-sft-v0.49/selection-plan.json'
CODE = FROZEN_CODE + ['scripts/prepare_attorney_choice_folds_v050.py', 'scripts/prepare_attorney_choice_sft_v049.py']


def take_up_to(pool, quotas, salt, scenario_cap):
    """The v0.49 `take` without its shortfall error, for the dev set only.

    The dev set only reports progress; it never selects an adapter, so a short cell is recorded instead of stopping.
    """
    chosen, shortfall = [], {}
    for mode in sorted(quotas):
        for name in sorted(quotas[mode]):
            quota, picked, per_scenario = quotas[mode][name], [], Counter()
            for row in sorted(pool.get((mode, name), []), key=lambda r: hash_key(salt, r['key'])):
                if quota is not None and len(picked) == quota:
                    break
                if quota is not None and per_scenario[row['scenario']] >= scenario_cap:
                    continue
                per_scenario[row['scenario']] += 1
                picked.append(row)
            if quota is not None and len(picked) < quota:
                shortfall[f'{mode}|{name}'] = {'quota': quota, 'taken': len(picked)}
            for position, row in enumerate(picked):
                chosen.append(dict(row, attorney_slot='A' if position % 2 == 0 else 'B'))
    return chosen, shortfall


def build(fold):
    plan = json.loads(PLAN.read_text())
    limits = json.loads((ROOT / ROUNDS[0][0] / 'selection-plan.json').read_text())['limits']
    rows, _, sources = build_population(limits)
    split, balance = plan['split'], plan['balance']
    folds = assign_folds(sorted({r['scenario'] for r in rows}), split['folds'], split['salt'])
    test_inputs, test_annotations, test_rows = load_test(folds, fold)
    test_originals = {r['original'] for r in test_rows}
    train_side = [r for r in rows if folds[r['scenario']] != fold]
    dev_side = [r for r in train_side if is_dev(r['original'], balance['salt'], balance['dev_share'])]
    fit_side = [r for r in train_side if not is_dev(r['original'], balance['salt'], balance['dev_share'])]
    modes = plan['distractors']['training']
    limit = balance['max_user_message_chars']
    fit_pool, dev_pool = candidates(fit_side, test_originals, modes, limit), candidates(dev_side, test_originals, modes, limit)
    cap = balance['scenario_cap_for_counted_cells']
    train = take(fit_pool, balance['train_quotas'], balance['salt'], cap)
    dev, dev_shortfall = take_up_to(dev_pool, balance['dev_quotas'], balance['salt'], cap)
    assert_disjoint(train + dev, test_rows)
    assert not {r['original'] for r in train} & {r['original'] for r in dev}
    assert 'targeted' not in {r['mode'] for r in train + dev}
    order = lambda data: sorted(data, key=lambda r: hash_key(balance['salt'] + 'order/', r['key']))
    train, dev = order(train), order(dev)
    summary = {'status': 'prepared_before_student_outputs', 'fold': fold, 'folds': folds, 'selection_plan_sha256': sha(PLAN),
               'test_cases': len(test_inputs), 'test_by_cell': dict(Counter(a['cell'] for a in test_annotations)),
               'test_astra_correct': sum(a['astra_correct'] for a in test_annotations),
               'train_rows': len(train), 'dev_rows': len(dev), 'dev_shortfall': dev_shortfall,
               'train_by_mode_cell': dict(Counter(r['mode'] + '|' + r['cell'] for r in train)),
               'train_more_chars': sum(r['cell'].startswith('more_chars') for r in train),
               'train_fewer_ops': sum(r['cell'].endswith('/fewer_ops') for r in train),
               'train_slots': dict(Counter(r['attorney_slot'] for r in train)),
               'train_by_scenario': dict(Counter(r['scenario'] for r in train)),
               'source_sha256': sources}
    return summary, train, dev, test_inputs, test_annotations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, choices=(1, 2), required=True)
    parser.add_argument('--dry-run', action='store_true', help='Print the counts, write nothing')
    args = parser.parse_args()
    out = ROOT / f'datasets/attorney-choice-sft-v0.50-fold{args.fold}'
    assert not (out / 'protocol.json').exists(), 'Already frozen'
    summary, train, dev, test_inputs, test_annotations = build(args.fold)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('source_sha256', 'folds', 'train_by_scenario')}, ensure_ascii=False, indent=1))
    if args.dry_run:
        print('Dry run: nothing written.')
        return
    (out / 'test').mkdir(parents=True, exist_ok=True)
    for name, data in [('train', train), ('dev', dev)]:
        (out / f'{name}.jsonl').write_text(''.join(json.dumps(
            training_example(r['attorney'], r['distractor'], r['attorney_slot']), ensure_ascii=False) + '\n' for r in data))
        (out / f'{name}-index.jsonl').write_text(''.join(json.dumps(
            {k: r[k] for k in ('mode', 'cell', 'scenario', 'attorney_slot')}, ensure_ascii=False) + '\n' for r in data))
    (out / 'test/inputs.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in test_inputs))
    (out / 'test/annotations.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in test_annotations))
    (out / 'extraction.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (out / 'README.md').write_text(README.replace('v0.49', f'v0.50, katman {args.fold}'))
    frozen = {name: sha(out / name) for name in ['README.md', 'extraction.json', 'train.jsonl', 'dev.jsonl',
                                                 'train-index.jsonl', 'dev-index.jsonl', 'test/inputs.jsonl', 'test/annotations.jsonl']}
    frozen.update({'../../' + name: sha(ROOT / name) for name in sorted(set(CODE))})
    frozen['../attorney-choice-sft-v0.49/selection-plan.json'] = sha(PLAN)
    protocol = {'version': f'attorney-choice-sft-v0.50-fold{args.fold}', 'frozen_at': datetime.now(timezone.utc).isoformat(),
                'frozen_sha256': frozen, 'student_calls_at_freeze': 0, 'difficulty_gate_round': False}
    (out / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
    print('Donduruldu:', (out / 'protocol.json').relative_to(ROOT))


if __name__ == '__main__':
    main()
