import argparse
import json
from pathlib import Path
from grounding_core import Interval, grounded_correct, temporal_iou

def to_interval(value):
    return Interval(value["start"], value["end"], value.get("score", 1.0), value["speaker"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.records).read_text(encoding="utf-8").splitlines()]
    ious, grounded = [], []
    for row in rows:
        pred, ref = to_interval(row["prediction"]), to_interval(row["reference"])
        ious.append(temporal_iou(pred, ref))
        grounded.append(grounded_correct(pred, ref, row["answer_correct"], args.iou_threshold))
    print(json.dumps({"samples": len(rows), "mean_iou": sum(ious) / len(ious), "grounded_accuracy": sum(grounded) / len(grounded)}, indent=2))

if __name__ == "__main__":
    main()
