"""Score v0.51 runs with balanced accuracy, build the heuristic baseline, and compare two runs.

  python3 -m scripts.score_surgical_verifier_v051 --run <dir> --name "<ad>"
  python3 -m scripts.score_surgical_verifier_v051 --heuristic --out <dir>
  python3 -m scripts.score_surgical_verifier_v051 --compare <dir-a> <dir-b>
"""
import argparse
import json
from pathlib import Path
import random

from open_surgeon.core import measure, redline
from scripts.attorney_choice_v046 import changed_chars, endpoint
from scripts.surgical_verifier_v051 import balanced_accuracy, bootstrap_sd, parse_verdict
from scripts.validate_contract_challenge import read, sha

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets/attorney-surgical-verifier-v0.51'


def check_frozen():
    """Pinned files must match their hash, or be listed in provenance.json as rewritten for release or not included."""
    provenance = json.loads((ROOT / 'provenance.json').read_text())
    for name, digest in json.loads((DATA / 'protocol.json').read_text())['frozen_sha256'].items():
        path = (DATA / name).resolve()
        rel = str(path.relative_to(ROOT))
        if path.exists() and sha(path) == digest:
            continue
        entry = provenance['rewritten'].get(rel)
        if entry and entry['original_sha256'] == digest and path.exists() and sha(path) == entry['release_sha256']:
            continue
        assert rel in provenance['not_included'] and not path.exists(), name


def rows_for(run):
    annotations = read(DATA / 'test/annotations.jsonl')
    predictions = {row['id']: row for row in read(run / 'predictions.jsonl')}
    assert set(predictions) == {row['id'] for row in annotations}, 'Predictions must cover every test item once'
    return [dict(row, verdict=parse_verdict(predictions[row['id']]['output'])) for row in annotations]


def score(run, name):
    rows = rows_for(run)
    result = balanced_accuracy(rows)
    result['bootstrap_sd'] = bootstrap_sd(rows, {row['id']: row['group'] for row in rows})
    by_cell = {}
    for cell in sorted({row['cell'] for row in rows}):
        result_cell = balanced_accuracy([row for row in rows if row['cell'] == cell])
        by_cell[cell] = {k: result_cell[k] for k in ('balanced_accuracy', 'recall_yes', 'recall_no', 'n_yes', 'n_no')}
    result.update(name=name, by_cell=by_cell, predictions_sha256=sha(run / 'predictions.jsonl'),
                  cases=[{k: row[k] for k in ('id', 'label', 'verdict', 'cell', 'group')} for row in rows])
    (run / 'score.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    pct = lambda value: f'{100 * value:.1f}%'
    lines = [f'# {name}: Crosby-style surgicality test', '',
             f"Balanced accuracy on {result['n_yes'] + result['n_no']} questions: **{pct(result['balanced_accuracy'])}** (±{100 * result['bootstrap_sd']:.1f} points, paragraph bootstrap). "
             f"Recall on yes {pct(result['recall_yes'])}, recall on no {pct(result['recall_no'])}. Invalid answers: {result['invalid']}.", '',
             'The labels are not attorney-reviewed; the numbers are not comparable with Crosby\'s attorney-labelled test.', '']
    (run / 'REPORT.md').write_text('\n'.join(lines))
    return result


def heuristic_score(item_messages):
    ops = json.loads(item_messages[1]['content'])['redline']
    ops = [{'kind': op['kind'], 'text': op['text']} for op in ops]
    source, target = endpoint(ops, False), endpoint(ops, True)
    return measure(source, target, ops, changed_chars(redline(source, target, 'targeted')))['score']


def heuristic(out):
    """Threshold fixed on training items only, then applied once to the test."""
    labels = {row['id']: row['label'] for row in read(DATA / 'train/labels.jsonl')}
    train = [(heuristic_score(row['messages']), labels[row['id']]) for row in read(DATA / 'train/prompts.jsonl')]
    best = None
    for threshold in sorted({value for value, _ in train}):
        rows = [{'label': label, 'verdict': 'yes' if value >= threshold else 'no'} for value, label in train]
        value = balanced_accuracy(rows)['balanced_accuracy']
        if best is None or value > best[1]:
            best = (threshold, value)
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'predictions.jsonl').open('w') as handle:
        for row in read(DATA / 'development/inputs.jsonl'):
            verdict = 'yes' if heuristic_score(row['messages']) >= best[0] else 'no'
            handle.write(json.dumps({'id': row['id'], 'output': json.dumps({'surgical': verdict})}) + '\n')
    (out / 'protocol.json').write_text(json.dumps({'method': 'open_surgeon.core.measure score >= threshold', 'threshold': best[0],
                                                   'train_balanced_accuracy': best[1], 'dataset_protocol_sha256': sha(DATA / 'protocol.json')}, indent=2) + '\n')
    return score(out, 'Heuristic (core.measure)')


def compare(first, second, draws=2000, seed=42):
    """Share of paragraph-bootstrap draws in which the first run's balanced accuracy is higher."""
    a, b = rows_for(first), rows_for(second)
    by_id = {row['id']: row for row in b}
    groups = sorted({row['group'] for row in a})
    members = {group: [row['id'] for row in a if row['group'] == group] for group in groups}
    by_a = {row['id']: row for row in a}
    rng, wins = random.Random(seed), 0
    for _ in range(draws):
        ids = [i for group in rng.choices(groups, k=len(groups)) for i in members[group]]
        diff = balanced_accuracy([by_a[i] for i in ids])['balanced_accuracy'] - balanced_accuracy([by_id[i] for i in ids])['balanced_accuracy']
        wins += diff > 0
    return {'first': str(first), 'second': str(second), 'first_minus_second': balanced_accuracy(a)['balanced_accuracy'] - balanced_accuracy(b)['balanced_accuracy'],
            'share_of_draws_first_higher': wins / draws}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path)
    parser.add_argument('--name')
    parser.add_argument('--heuristic', action='store_true')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--compare', type=Path, nargs=2)
    args = parser.parse_args()
    check_frozen()
    if args.compare:
        print(json.dumps(compare(*args.compare), indent=1))
        return
    result = heuristic(args.out) if args.heuristic else score(args.run, args.name)
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
