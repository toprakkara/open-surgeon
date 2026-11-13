"""Score runs on the v0.52 agent-redline test, apply the v0.51 heuristic threshold unchanged, and compare runs.

  python3 -m scripts.score_surgical_agent_v052 --run <dir> --name "<ad>"
  python3 -m scripts.score_surgical_agent_v052 --heuristic --out <dir>
  python3 -m scripts.score_surgical_agent_v052 --compare <dir-a> <dir-b>
"""
import argparse
import json
from pathlib import Path

import scripts.score_surgical_verifier_v051 as base
from scripts.validate_contract_challenge import read, sha

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets/attorney-surgical-agent-v0.52'
V051_HEURISTIC = ROOT / 'training/surgical-verifier-v051/heuristic/protocol.json'


def use_dataset():
    """The v0.51 scorer reads its module-level DATA; point it at v0.52 for every call below."""
    base.DATA = DATA
    base.check_frozen()


def heuristic(out):
    threshold = json.loads(V051_HEURISTIC.read_text())['threshold']
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'predictions.jsonl').open('w') as handle:
        for row in read(DATA / 'development/inputs.jsonl'):
            verdict = 'yes' if base.heuristic_score(row['messages']) >= threshold else 'no'
            handle.write(json.dumps({'id': row['id'], 'output': json.dumps({'surgical': verdict})}) + '\n')
    (out / 'protocol.json').write_text(json.dumps({'method': 'open_surgeon.core.measure score >= threshold fixed in v0.51', 'threshold': threshold,
                                                   'threshold_source_sha256': sha(V051_HEURISTIC), 'dataset_protocol_sha256': sha(DATA / 'protocol.json')}, indent=2) + '\n')
    return base.score(out, 'Heuristic (core.measure, v0.51 threshold)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path)
    parser.add_argument('--name')
    parser.add_argument('--heuristic', action='store_true')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--compare', type=Path, nargs=2)
    args = parser.parse_args()
    use_dataset()
    if args.compare:
        print(json.dumps(base.compare(*args.compare), indent=1))
        return
    result = heuristic(args.out) if args.heuristic else base.score(args.run, args.name)
    result.pop('cases')
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
