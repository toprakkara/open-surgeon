"""Build the fold-0 training data and test cases before any student output is seen."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json

from open_surgeon.core import redline
from scripts.attorney_choice_sft_v049 import assert_disjoint, assign_folds, training_example
from scripts.attorney_choice_v046 import build_input, canonical_ops, cell, hash_key, only_whitespace_boundary
from scripts.extract_redlinebench_paragraphs import ROOT
from scripts.prepare_attorney_choice_v046 import build_population
from scripts.validate_contract_challenge import canon, read, sha

OUT = ROOT / 'datasets/attorney-choice-sft-v0.49'
ROUNDS = [('datasets/attorney-markup-choice-v0.46', 'training/astra-attorney-choice-v046'),
          ('datasets/attorney-markup-balanced-v0.47', 'training/astra-attorney-balanced-v047')]
HINT = 'training/astra-attorney-hint-control-v047/score.json'
FROZEN_CODE = ['scripts/attorney_choice_sft_v049.py', 'scripts/prepare_attorney_choice_sft_v049.py',
               'scripts/score_attorney_choice_sft_v049.py', 'scripts/evaluate_attorney_choice_local_v049.py',
               'scripts/attorney_choice_v046.py', 'scripts/prepare_attorney_choice_v046.py']
README = '''# Attorney markup training data v0.49

Trains a small model on the question "which presentation is the attorney's markup?". The result measures **likeness to
the attorney, not surgicality**. Test scenarios and paragraphs from the same source are kept out of training. The wrong
options in training come from the `word`, `character` and `whole` diffs; the `targeted` diff used in the test is not
used. Because `word` and `targeted` coincide on most paragraphs, the style of the diff is not fully hidden.

Source: [ContractBench](https://huggingface.co/datasets/micro1-inc/ContractBench), CC BY 4.0.
Changes: paragraphs were projected, change blocks were normalised, automatic diffs were added.
'''


def load_test(folds, fold):
    hint = {c['id']: c for c in json.loads((ROOT / HINT).read_text())['cases']}
    inputs, annotations, rows = [], [], []
    for data, run in ROUNDS:
        answers = {c['id']: c for c in json.loads((ROOT / run / 'score.json').read_text())['cases']}
        by_id = {r['id']: r for r in read(ROOT / data / 'development/inputs.jsonl')}
        for row in read(ROOT / data / 'selected.jsonl'):
            if folds[row['scenario']] != fold:
                continue
            answer = answers[row['id']]
            inputs.append(by_id[row['id']])
            rows.append(row)
            annotations.append({'id': row['id'], 'attorney_slot': row['attorney_slot'], 'cell': row['cell'],
                                'scenario': row['scenario'], 'rules': row['rules'], 'source_round': data.split('/')[1],
                                'astra_correct': answer['correct'],
                                'astra_hint_correct': hint[row['id']]['correct'] if row['id'] in hint else None})
    return inputs, annotations, rows


def candidates(rows, test_originals, modes, limit):
    markups = defaultdict(set)
    for row in rows:
        markups[(row['scenario'], row['original'], row['revision'])].add(row['key'])
    pool = defaultdict(list)
    for row in rows:
        if row['original'] in test_originals:
            continue
        for mode in modes:
            distractor = canonical_ops(redline(row['original'], row['revision'], mode))
            if (distractor == row['attorney'] or only_whitespace_boundary(row['attorney'], distractor)
                    or canon(distractor) in markups[(row['scenario'], row['original'], row['revision'])]):
                continue
            if len(build_input(row['attorney'], distractor, 'A')['messages'][1]['content']) > limit:
                continue
            pool[(mode, cell(row['attorney'], distractor))].append({
                'key': row['key'] + '/' + mode, 'mode': mode, 'cell': cell(row['attorney'], distractor),
                'scenario': row['scenario'], 'original': row['original'], 'attorney': row['attorney'], 'distractor': distractor})
    return pool


def take(pool, quotas, salt, scenario_cap):
    chosen = []
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
                raise SystemExit(f'{mode} {name}: {len(picked)} of {quota}')
            for position, row in enumerate(picked):
                chosen.append(dict(row, attorney_slot='A' if position % 2 == 0 else 'B'))
    return chosen


def is_dev(original, salt, share):
    return int(hash_key(salt + 'dev/', original)[:8], 16) / 16 ** 8 < share


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Print the counts, write nothing')
    args = parser.parse_args()
    assert not (OUT / 'protocol.json').exists(), 'Already frozen'
    plan = json.loads((OUT / 'selection-plan.json').read_text())
    limits = json.loads((ROOT / ROUNDS[0][0] / 'selection-plan.json').read_text())['limits']
    rows, _, sources = build_population(limits)
    split, balance = plan['split'], plan['balance']
    folds = assign_folds(sorted({r['scenario'] for r in rows}), split['folds'], split['salt'])
    fold = split['first_fold_tested']
    test_inputs, test_annotations, test_rows = load_test(folds, fold)
    test_originals = {r['original'] for r in test_rows}
    train_side = [r for r in rows if folds[r['scenario']] != fold]
    dev_side = [r for r in train_side if is_dev(r['original'], balance['salt'], balance['dev_share'])]
    fit_side = [r for r in train_side if not is_dev(r['original'], balance['salt'], balance['dev_share'])]
    modes = plan['distractors']['training']
    limit = balance['max_user_message_chars']
    fit_pool, dev_pool = candidates(fit_side, test_originals, modes, limit), candidates(dev_side, test_originals, modes, limit)
    print('training pool:', {f'{m}|{c}': len(v) for (m, c), v in sorted(fit_pool.items())})
    print('dev havuzu:', {f'{m}|{c}': len(v) for (m, c), v in sorted(dev_pool.items())})
    cap = balance['scenario_cap_for_counted_cells']
    train = take(fit_pool, balance['train_quotas'], balance['salt'], cap)
    dev = take(dev_pool, balance['dev_quotas'], balance['salt'], cap)
    assert_disjoint(train + dev, test_rows)
    assert not {r['original'] for r in train} & {r['original'] for r in dev}
    assert 'targeted' not in {r['mode'] for r in train + dev}
    order = lambda data: sorted(data, key=lambda r: hash_key(balance['salt'] + 'order/', r['key']))
    train, dev = order(train), order(dev)
    summary = {'status': 'prepared_before_student_outputs', 'fold': fold, 'folds': folds,
               'test_cases': len(test_inputs), 'test_by_cell': dict(Counter(a['cell'] for a in test_annotations)),
               'test_astra_correct': sum(a['astra_correct'] for a in test_annotations),
               'train_rows': len(train), 'dev_rows': len(dev),
               'train_by_mode_cell': dict(Counter(r['mode'] + '|' + r['cell'] for r in train)),
               'train_more_chars': sum(r['cell'].startswith('more_chars') for r in train),
               'train_fewer_ops': sum(r['cell'].endswith('/fewer_ops') for r in train),
               'train_slots': dict(Counter(r['attorney_slot'] for r in train)),
               'train_by_scenario': dict(Counter(r['scenario'] for r in train)),
               'source_sha256': sources}
    print(json.dumps({k: v for k, v in summary.items() if k not in ('source_sha256', 'folds')}, ensure_ascii=False, indent=1))
    if args.dry_run:
        print('Dry run: nothing written.')
        return
    for name, data in [('train', train), ('dev', dev)]:
        (OUT / f'{name}.jsonl').write_text(''.join(json.dumps(
            training_example(r['attorney'], r['distractor'], r['attorney_slot']), ensure_ascii=False) + '\n' for r in data))
        (OUT / f'{name}-index.jsonl').write_text(''.join(json.dumps(
            {k: r[k] for k in ('mode', 'cell', 'scenario', 'attorney_slot')}, ensure_ascii=False) + '\n' for r in data))
    (OUT / 'test').mkdir(exist_ok=True)
    (OUT / 'test/inputs.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in test_inputs))
    (OUT / 'test/annotations.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in test_annotations))
    (OUT / 'extraction.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'README.md').write_text(README)
    frozen = {name: sha(OUT / name) for name in ['README.md', 'selection-plan.json', 'extraction.json', 'train.jsonl', 'dev.jsonl',
                                                 'train-index.jsonl', 'dev-index.jsonl', 'test/inputs.jsonl', 'test/annotations.jsonl']}
    frozen.update({'../../' + name: sha(ROOT / name) for name in FROZEN_CODE})
    protocol = {'version': plan['version'], 'frozen_at': datetime.now(timezone.utc).isoformat(), 'frozen_sha256': frozen,
                'student_calls_at_freeze': 0, 'difficulty_gate_round': False}
    (OUT / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
    print('Donduruldu:', (OUT / 'protocol.json').relative_to(ROOT))


if __name__ == '__main__':
    main()
