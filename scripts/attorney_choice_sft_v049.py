"""Pure helpers for training a small model on the attorney-markup choice task."""
from collections import defaultdict
import json
import math

from scripts.attorney_choice_v046 import build_input, hash_key


def mcnemar_exact(only_first_right, only_second_right):
    """Two-sided exact McNemar p-value from the two discordant counts."""
    total = only_first_right + only_second_right
    if not total:
        return 1.0
    smaller = min(only_first_right, only_second_right)
    tail = sum(math.comb(total, k) for k in range(smaller + 1)) / 2 ** total
    return min(1.0, 2 * tail)


def paired_counts(first, second):
    """Count paired outcomes from two id-to-correct maps covering the same cases."""
    if set(first) != set(second):
        raise ValueError('Models must cover the same cases')
    return {'both_right': sum(first[key] and second[key] for key in first),
            'only_first_right': sum(first[key] and not second[key] for key in first),
            'only_second_right': sum(second[key] and not first[key] for key in first),
            'both_wrong': sum(not first[key] and not second[key] for key in first)}


def assign_folds(scenarios, folds, salt):
    """Deal scenarios to folds by contract family, so each fold keeps a similar mix."""
    families = defaultdict(list)
    for scenario in scenarios:
        families[scenario.split('-')[0]].append(scenario)
    result, position = {}, 0
    for family in sorted(families):
        for scenario in sorted(families[family], key=lambda name: hash_key(salt, name)):
            result[scenario] = position % folds
            position += 1
    return result


def training_example(attorney, distractor, attorney_slot):
    """The target is only the choice; the prompt matches what Astra received."""
    messages = build_input(attorney, distractor, attorney_slot)['messages']
    return {'messages': messages + [{'role': 'assistant', 'content': json.dumps({'choice': attorney_slot})}]}


def assert_disjoint(train_rows, test_rows):
    """No scenario and no exact source paragraph may appear on both sides."""
    shared_scenarios = {row['scenario'] for row in train_rows} & {row['scenario'] for row in test_rows}
    shared_text = {row['original'] for row in train_rows} & {row['original'] for row in test_rows}
    if shared_scenarios or shared_text:
        raise ValueError(f'Train and test overlap: {len(shared_scenarios)} scenarios, {len(shared_text)} paragraphs')
