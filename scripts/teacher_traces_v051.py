"""Make blind teacher batches, read teacher answers back, and keep the traces whose verdict matches the label.

  python3 -m scripts.teacher_traces_v051 check <answers.txt> [...]              verdict accuracy against the training labels
  python3 -m scripts.teacher_traces_v051 batches --size 25 --skip <answers.txt>  write blind batch files for the remaining items
  python3 -m scripts.teacher_traces_v051 keep <answers.txt> [...] --out <jsonl>  rejection sampling into SFT examples
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from scripts.attorney_choice_v046 import hash_key
from scripts.surgical_verifier_v051 import SYSTEM, parse_verdict
from scripts.validate_contract_challenge import read, sha

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets/attorney-surgical-verifier-v0.51'
TEACHER = ROOT / 'training/surgical-verifier-v051/teacher'
SECTION = re.compile(r'^### ([0-9a-f]{64})\s*$', re.M)
# Teacher copying slips, fixed here so the raw answer files stay untouched. Each one is the answer in the same
# position as the item it belongs to and differs from that item's id by one character.
ID_FIXES = {
    'd55da66ab63dfae4378079c5f7ca0b2ccb94c788c4c62944d3b282d055a36c20':
        'd55da66ab63dfae4378079c5f7ca0b2ccb94c788c6c62944d3b282d055a36c20',  # batch-013, position 40
}


def parse_answers(path):
    text = Path(path).read_text()
    marks = list(SECTION.finditer(text))
    answers = {}
    for mark, following in zip(marks, marks[1:] + [None]):
        body = text[mark.end():following.start() if following else len(text)].strip()
        key = ID_FIXES.get(mark.group(1), mark.group(1))
        assert key not in answers, f'Repeated id in {path}'
        answers[key] = body
    return answers


def words(body):
    return len(body.rsplit('\n', 1)[0].split())


def check(paths):
    labels = {row['id']: row for row in read(DATA / 'train/labels.jsonl')}
    for path in paths:
        answers = parse_answers(path)
        rows = [(labels[key], parse_verdict(body), words(body)) for key, body in answers.items()]
        right = sum(verdict == meta['label'] for meta, verdict, _ in rows)
        by_label = {label: f"{sum(v == label for m, v, _ in rows if m['label'] == label)}/{sum(m['label'] == label for m, _, _ in rows)}" for label in ('yes', 'no')}
        print(json.dumps({'file': str(path), 'answers': len(rows), 'correct': right, 'invalid': sum(v is None for _, v, _ in rows),
                          'recall_by_label': by_label, 'words_min_max': [min(w for *_, w in rows), max(w for *_, w in rows)]}, ensure_ascii=False))


def batches(size, skip):
    """Both items from one source paragraph never share a batch, so the teacher cannot compare them."""
    done = set()
    for path in skip:
        done |= set(parse_answers(path))
    labels = {row['id']: row for row in read(DATA / 'train/labels.jsonl')}
    prompts = [row for row in read(DATA / 'train/prompts.jsonl') if row['id'] not in done]
    count = -(-len(prompts) // size)
    buckets, by_group = [[] for _ in range(count)], defaultdict(list)
    for row in prompts:
        by_group[labels[row['id']]['group']].append(row)
    cursor = 0
    for group in sorted(by_group, key=lambda g: hash_key('teacher-batches/', g)):
        members = by_group[group]
        assert len(members) <= count, 'Too few batches to separate a paragraph'
        for offset, row in enumerate(members):
            buckets[(cursor + offset) % count].append(row)
        cursor += len(members)
    folder = TEACHER / 'batches'
    folder.mkdir(parents=True, exist_ok=True)
    for index, bucket in enumerate(buckets):
        with (folder / f'batch-{index:03d}.jsonl').open('w') as handle:
            for row in bucket:
                handle.write(json.dumps({'id': row['id'], 'redline': json.loads(row['messages'][1]['content'])['redline']}, ensure_ascii=False) + '\n')
    print(json.dumps({'items': len(prompts), 'batches': count, 'sizes': dict(Counter(len(b) for b in buckets))}))


def keep(paths, out):
    labels = {row['id']: row for row in read(DATA / 'train/labels.jsonl')}
    prompts = {row['id']: row for row in read(DATA / 'train/prompts.jsonl')}
    answers = {}
    for path in paths:
        for key, body in parse_answers(path).items():
            assert key not in answers, f'Two teacher answers for {key}'
            answers[key] = body
    kept, stats = defaultdict(list), Counter()
    for key in sorted(answers, key=lambda k: hash_key('keep-order/', k)):
        verdict = parse_verdict(answers[key])
        status = 'invalid' if verdict is None else ('kept' if verdict == labels[key]['label'] else 'wrong')
        stats[f"{labels[key]['label']}|{status}"] += 1
        if status == 'kept':
            assert prompts[key]['messages'][0]['content'] == SYSTEM
            kept[labels[key]['label']].append({'messages': prompts[key]['messages'] + [{'role': 'assistant', 'content': answers[key]}]})
    # Class balance from the plan: equal yes and no, trimmed in the salted order above.
    # --no-balance keeps everything, for v0.52, which balances after pooling two datasets.
    size = min(len(kept['yes']), len(kept['no']))
    balanced = kept['yes'][:size] + kept['no'][:size] if BALANCE else kept['yes'] + kept['no']
    balanced.sort(key=lambda row: hash_key('train-order/', row['messages'][1]['content']))
    Path(out).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in balanced))
    summary = {'answers': len(answers), 'kept_before_balance': {k: len(v) for k, v in kept.items()}, 'kept': len(balanced),
               'by_label_status': dict(stats), 'sources': {str(p): sha(Path(p)) for p in paths}, 'out_sha256': sha(Path(out))}
    Path(out).with_suffix('.summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=1))


BALANCE = True


def main():
    global DATA, TEACHER, BALANCE
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, help='Another frozen dataset with train/prompts.jsonl and train/labels.jsonl')
    parser.add_argument('--teacher', type=Path, help='Folder for batches and answers of that dataset')
    parser.add_argument('--no-balance', action='store_true')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('check').add_argument('paths', nargs='+')
    make = sub.add_parser('batches')
    make.add_argument('--size', type=int, default=25)
    make.add_argument('--skip', nargs='*', default=[])
    take = sub.add_parser('keep')
    take.add_argument('paths', nargs='+')
    take.add_argument('--out', required=True)
    args = parser.parse_args()
    DATA = ROOT / args.data if args.data else DATA
    TEACHER = ROOT / args.teacher if args.teacher else TEACHER
    BALANCE = not args.no_balance
    if args.command == 'check':
        check(args.paths)
    elif args.command == 'batches':
        batches(args.size, args.skip)
    else:
        keep(args.paths, args.out)


if __name__ == '__main__':
    main()
