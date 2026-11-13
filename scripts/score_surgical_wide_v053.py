"""Score runs on the v0.53 test with the v0.52 scorer, pointed at the v0.53 dataset.

  python3 -m scripts.score_surgical_wide_v053 --run <dir> --name "<ad>"
  python3 -m scripts.score_surgical_wide_v053 --heuristic --out <dir>
  python3 -m scripts.score_surgical_wide_v053 --compare <dir-a> <dir-b>
"""
from pathlib import Path

import scripts.score_surgical_agent_v052 as previous

previous.DATA = Path(__file__).resolve().parents[1] / 'datasets/attorney-surgical-wide-v0.53'

if __name__ == '__main__':
    previous.main()
