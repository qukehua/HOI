"""Compare one identical H/O/C mode only after checking the evaluation protocol."""
import argparse
import json
from pathlib import Path

from unified_hoi.parametric.evaluation_protocol import assert_comparable, MODE_DESCRIPTIONS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--left", required=True, help="Left summary.json")
    p.add_argument("--right", required=True, help="Right summary.json")
    p.add_argument("--mode", choices=MODE_DESCRIPTIONS, required=True)
    args = p.parse_args()
    left = json.loads(Path(args.left).read_text(encoding="utf-8"))
    right = json.loads(Path(args.right).read_text(encoding="utf-8"))
    print(json.dumps(assert_comparable(left, right, args.mode), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
