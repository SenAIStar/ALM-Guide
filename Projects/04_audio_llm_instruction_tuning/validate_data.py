import argparse
import json
from pathlib import Path
from audio_llm_core import validate_instruction_sample

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    rows = [validate_instruction_sample(json.loads(line)) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()]
    print(json.dumps({"samples": len(rows), "tasks": sorted({row["task"] for row in rows})}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
