import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.records).read_text(encoding="utf-8").splitlines()]
    required = ["bleu", "asr_wer", "speaker_similarity", "rtf"]
    report = {key: sum(float(row[key]) for row in rows) / len(rows) for key in required}
    report["samples"] = len(rows)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
