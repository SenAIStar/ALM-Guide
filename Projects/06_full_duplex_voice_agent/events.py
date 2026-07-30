from dataclasses import dataclass, asdict
import json
import time

@dataclass(frozen=True)
class Event:
    session_id: str
    event: str
    timestamp_ms: int
    generation_id: str | None = None

    def to_json(self):
        return json.dumps(asdict(self), ensure_ascii=False)

def now_event(session_id, event, generation_id=None):
    return Event(session_id, event, time.monotonic_ns() // 1_000_000, generation_id)

def paired_latency(events, start_name, end_name):
    starts, values = {}, []
    for event in events:
        key = event.get("generation_id") or event["session_id"]
        if event["event"] == start_name:
            starts[key] = event["timestamp_ms"]
        elif event["event"] == end_name and key in starts:
            values.append(event["timestamp_ms"] - starts.pop(key))
    return values
