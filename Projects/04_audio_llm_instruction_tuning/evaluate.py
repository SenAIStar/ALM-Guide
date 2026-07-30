import argparse
import json
from pathlib import Path
from audio_llm_core import task_macro_average

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.records).read_text(encoding="utf-8").splitlines()]
    print(json.dumps(task_macro_average(rows), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
