import argparse
import json
from pathlib import Path
from events import paired_latency

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.events).read_text(encoding="utf-8").splitlines()]
    latency = paired_latency(rows, "barge_in_detected", "generation_cancelled")
    print(json.dumps({"cancel_samples": len(latency), "cancel_latency_ms_mean": sum(latency) / len(latency) if latency else None, "metrics_status": "measured_from_event_log"}, indent=2))

if __name__ == "__main__":
    main()
