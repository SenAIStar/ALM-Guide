import argparse
import json
from pathlib import Path

from asr_core import error_rate, normalize_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--unit", choices=["word", "char"], default="char")
    args = parser.parse_args()
    refs = Path(args.reference).read_text(encoding="utf-8").splitlines()
    hyps = Path(args.hypothesis).read_text(encoding="utf-8").splitlines()
    if len(refs) != len(hyps):
        raise ValueError("reference and hypothesis counts differ")
    scores = [error_rate(normalize_text(r), normalize_text(h), args.unit) for r, h in zip(refs, hyps)]
    print(json.dumps({"samples": len(scores), "error_rate_macro": sum(scores) / len(scores)}, indent=2))


if __name__ == "__main__":
    main()
