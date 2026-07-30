import argparse
import json
from pathlib import Path
from omni_core import validate_sample

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    rows = [
        validate_sample(json.loads(line))
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(json.dumps({"samples": len(rows), "modality_sets": sorted({"+".join(sorted(row["modalities"])) for row in rows})}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
