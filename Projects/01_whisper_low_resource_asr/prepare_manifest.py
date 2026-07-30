import argparse
import json
from pathlib import Path

from asr_core import ManifestRow, assert_speaker_disjoint, normalize_text


def read_manifest(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(ManifestRow.from_dict(json.loads(line)))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", required=True, choices=["train", "validation", "test"])
    parser.add_argument(
        "--check-against",
        action="append",
        default=[],
        metavar="MANIFEST",
        help="Existing manifest to include in the speaker-disjoint check; repeat as needed",
    )
    args = parser.parse_args()
    rows = []
    for line in Path(args.input).read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        item["split"] = args.split
        item["text"] = normalize_text(item["text"])
        rows.append(ManifestRow.from_dict(item))
    comparison_rows = [row for path in args.check_against for row in read_manifest(path)]
    assert_speaker_disjoint([*comparison_rows, *rows])
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(json.dumps(row.__dict__, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
