"""Select and freeze the attorney-markup choice pilot before any model call."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json

from open_surgeon.core import measure, redline
from scripts.attorney_choice_v046 import (SYSTEM, build_input, canonical_ops, cell, change_ops, changed_chars, endpoint,
                                          hash_key, only_whitespace_boundary, rule_pick)
from scripts.extract_redlinebench_paragraphs import NS, ROOT
from scripts.prepare_contractbench_native_v037 import paragraphs, project
from scripts.validate_contract_challenge import canon, read, sha

OUT = ROOT / 'datasets/attorney-markup-choice-v0.46'
POOL = ROOT / 'datasets/contractbench-source-pool'
FROZEN_CODE = ['scripts/attorney_choice_v046.py', 'scripts/prepare_attorney_choice_v046.py',
               'scripts/score_attorney_choice_v046.py', 'scripts/prepare_contractbench_native_v037.py',
               'scripts/extract_redlinebench_paragraphs.py']
README = '''# Attorney markup choice v0.46

Two presentations of the same change are given: the attorney's real markup and an automatic targeted diff. The model is
asked to find the attorney's. The label only says where a presentation came from; it does not say which is better. The
result measures **likeness to the attorney, not surgicality**.

- Rules, scoring and interpretation thresholds were fixed in `selection-plan.json` before any model output.
- Two simple rules (more characters, fewer operations) were cross-balanced. Each is clearly right on 24 of the 48 items
  and undecided on 13; with coin flips for ties the expected accuracy is about 64%.
- The items are not independent and come from fictional contracts; 48 items is a small pilot.
- This set is not part of the difficulty gate and is not used for training; no student model is run.

Source: [ContractBench](https://huggingface.co/datasets/micro1-inc/ContractBench), CC BY 4.0.
Changes: paragraphs were projected, change blocks were normalised, an automatic diff was added.
'''


def relative(path):
    return str(path.relative_to(ROOT))


def build_population(limits):
    exclusions, rows, sources = Counter(), {}, {}
    for reference in sorted((POOL / 'references').glob('*-t1-*.docx')):
        task = reference.stem
        scenario = task.rsplit('-t1-', 1)[0]
        baseline = POOL / 'tasks' / task / 'environment/app/contract.docx'
        sources[relative(reference)] = sha(reference)
        sources[relative(baseline)] = sha(baseline)
        clean = Counter()
        for paragraph in paragraphs(baseline):
            try:
                ops, _, _ = project(paragraph)
            except ValueError:
                continue
            if all(op['kind'] == 'equal' for op in ops):
                clean[endpoint(ops, False)] += 1
        for index, paragraph in enumerate(paragraphs(reference)):
            if not paragraph.xpath('.//w:ins | .//w:del', namespaces=NS):
                continue
            try:
                raw, mode, _ = project(paragraph)
            except ValueError:
                exclusions['unsupported_xml'] += 1
                continue
            original, revision = endpoint(raw, False), endpoint(raw, True)
            size = max(len(original), len(revision))
            if original == revision:
                reason = 'no_text_change'
            elif not any(op['kind'] == 'equal' and op['text'] for op in raw):
                reason = 'no_retained_context'
            elif size < limits['min_chars']:
                reason = 'shorter_than_min'
            elif size > limits['max_chars']:
                reason = 'longer_than_max'
            elif clean.get(original) != 1:
                reason = 'no_unique_clean_match'
            else:
                reason = None
            if reason:
                exclusions[reason] += 1
                continue
            attorney = canonical_ops(raw)
            key = canon(attorney)
            row = rows.setdefault((scenario, key), {
                'key': key, 'scenario': scenario, 'original': original, 'revision': revision, 'attorney': attorney,
                'raw_ids': set(), 'provenance': []})
            row['raw_ids'].add(hashlib.sha256(canon(raw).encode()).hexdigest())
            row['provenance'].append({'task': task, 'paragraph_index': index, 'projection_mode': mode,
                                      'reference': relative(reference), 'reference_sha256': sources[relative(reference)]})
    return list(rows.values()), exclusions, sources


def apply_filters(rows, exclusions, limits):
    markups = defaultdict(set)
    for row in rows:
        markups[(row['scenario'], row['original'], row['revision'])].add(row['key'])
    eligible = []
    for row in rows:
        algorithm = canonical_ops(redline(row['original'], row['revision'], 'targeted'))
        attorney = row['attorney']
        if attorney == algorithm:
            reason = 'same_as_algorithm'
        elif only_whitespace_boundary(attorney, algorithm):
            reason = 'whitespace_boundary_only'
        elif canon(algorithm) in markups[(row['scenario'], row['original'], row['revision'])]:
            reason = 'algorithm_matches_another_attorney'
        elif len(build_input(attorney, algorithm, 'A')['messages'][1]['content']) > limits['max_user_message_chars']:
            reason = 'user_message_too_long'
        else:
            reason = None
        if reason:
            exclusions[reason] += 1
            continue
        assert endpoint(algorithm, False) == row['original'] == endpoint(attorney, False)
        assert endpoint(algorithm, True) == row['revision'] == endpoint(attorney, True)
        reference = changed_chars(algorithm)
        row.update(algorithm=algorithm, cell=cell(attorney, algorithm), rules={
            'more_chars': rule_pick(changed_chars(attorney), reference),
            'fewer_ops': rule_pick(-change_ops(attorney), -change_ops(algorithm)),
            'core_measure': rule_pick(measure(row['original'], row['revision'], attorney, reference)['score'],
                                      measure(row['original'], row['revision'], algorithm, reference)['score'])})
        eligible.append(row)
    return eligible


def select(eligible, balance):
    cells = defaultdict(list)
    for row in eligible:
        cells[row['cell']].append(row)
    assert set(cells) <= set(balance['cells'])
    availability = {name: len(cells[name]) for name in balance['cells']}
    overall, selected = Counter(), []
    for name in sorted(balance['cells'], key=lambda c: (availability[c], c)):
        chosen, in_cell = [], Counter()
        for row in sorted(cells[name], key=lambda r: hash_key(balance['salt'], r['key'])):
            if len(chosen) == balance['per_cell']:
                break
            scenario = row['scenario']
            if in_cell[scenario] >= balance['max_per_scenario_in_cell'] or overall[scenario] >= balance['max_per_scenario_overall']:
                continue
            in_cell[scenario] += 1
            overall[scenario] += 1
            chosen.append(row)
        if len(chosen) < balance['per_cell']:
            raise SystemExit(f'{name}: only {len(chosen)} items could be selected; review the plan before any model call.')
        for position, row in enumerate(chosen):
            row['attorney_slot'] = 'A' if position % 2 == 0 else 'B'
        selected.extend(chosen)
    return selected, availability


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Print the counts, write nothing')
    args = parser.parse_args()
    assert not (OUT / 'protocol.json').exists(), 'Already frozen'
    plan = json.loads((OUT / 'selection-plan.json').read_text())
    rows, exclusions, sources = build_population(plan['limits'])
    eligible = apply_filters(rows, exclusions, plan['limits'])
    selected, availability = select(eligible, plan['balance'])
    prior = {r['id'] for path in (ROOT / 'datasets').glob('*/development/inputs.jsonl') for r in read(path)}
    inputs, annotations, stored = [], [], []
    for row in selected:
        built = build_input(row['attorney'], row['algorithm'], row['attorney_slot'])
        inputs.append(built)
        annotations.append({'id': built['id'], 'attorney_slot': row['attorney_slot'], 'cell': row['cell'],
                            'scenario': row['scenario'], 'rules': row['rules']})
        stored.append({'id': built['id'], 'scenario': row['scenario'], 'cell': row['cell'],
                       'attorney_slot': row['attorney_slot'], 'original': row['original'], 'revision': row['revision'],
                       'attorney_operations': row['attorney'], 'algorithm_operations': row['algorithm'],
                       'rules': row['rules'], 'provenance': row['provenance'],
                       'attorney_markup_seen_in_prior_round': bool(row['raw_ids'] & prior)})
    order = sorted(range(len(inputs)), key=lambda i: hash_key(plan['balance']['salt'] + 'order/', inputs[i]['id']))
    inputs = [inputs[i] for i in order]
    annotations = [annotations[i] for i in order]
    stored = [stored[i] for i in order]
    assert len({r['id'] for r in inputs}) == len(inputs) == plan['balance']['per_cell'] * len(plan['balance']['cells'])
    rule_summary = {name: Counter(row['rules'][name] for row in stored) for name in ('more_chars', 'fewer_ops', 'core_measure')}
    extraction = {
        'status': 'selected_before_target_calls', 'target_calls': 0, 'eligible_pairs': len(eligible),
        'cell_availability': availability, 'exclusions': dict(exclusions),
        'selected_by_cell': dict(Counter(r['cell'] for r in stored)),
        'selected_by_scenario': dict(Counter(r['scenario'] for r in stored)),
        'attorney_slots': dict(Counter(r['attorney_slot'] for r in stored)),
        'simple_rules_on_selection': {name: dict(count) for name, count in rule_summary.items()},
        'attorney_markup_seen_in_prior_rounds': sum(r['attorney_markup_seen_in_prior_round'] for r in stored),
        'distractor_code_sha256_at_selection': sha(ROOT / 'open_surgeon/core.py'), 'source_sha256': sources}
    print(json.dumps({k: v for k, v in extraction.items() if k != 'source_sha256'}, ensure_ascii=False, indent=1))
    if args.dry_run:
        print('Dry run: nothing written.')
        return
    (OUT / 'development').mkdir(exist_ok=True)
    (OUT / 'labels').mkdir(exist_ok=True)
    for path, data in [(OUT / 'development/inputs.jsonl', inputs), (OUT / 'labels/annotations.jsonl', annotations),
                       (OUT / 'selected.jsonl', stored)]:
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in data))
    (OUT / 'extraction.json').write_text(json.dumps(extraction, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'system-prompt.txt').write_text(SYSTEM)
    (OUT / 'README.md').write_text(README)
    frozen = {name: sha(OUT / name) for name in ['README.md', 'selection-plan.json', 'extraction.json', 'selected.jsonl',
                                                 'system-prompt.txt', 'development/inputs.jsonl', 'labels/annotations.jsonl']}
    frozen.update({'../../' + name: sha(ROOT / name) for name in FROZEN_CODE})
    protocol = {'version': plan['version'], 'frozen_at': datetime.now(timezone.utc).isoformat(), 'n_inputs': len(inputs),
                'frozen_sha256': frozen, 'target_calls_at_freeze': 0, 'model': plan['model']['name'],
                'reasoning_effort': plan['model']['reasoning_effort'], 'difficulty_gate_round': False,
                'student_work': False, 'training_allowed': False}
    (OUT / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
    print('Donduruldu:', relative(OUT / 'protocol.json'))


if __name__ == '__main__':
    main()
