import argparse
import json
from pathlib import Path
from omni_core import counterfactual_report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.records).read_text(encoding="utf-8").splitlines()]
    reports = [counterfactual_report(row["full_score"], row["ablated_scores"], set(row["required_modalities"])) for row in rows]
    print(json.dumps({"samples": len(reports), "reports": reports}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
