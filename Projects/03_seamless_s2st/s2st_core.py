from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Chunk:
    index: int
    start: float
    end: float

def make_chunks(duration: float, chunk_seconds: float, overlap_seconds: float):
    if not 0 <= overlap_seconds < chunk_seconds:
        raise ValueError("overlap must be in [0, chunk)")
    chunks, start, index = [], 0.0, 0
    while start < duration:
        end = min(start + chunk_seconds, duration)
        chunks.append(Chunk(index, start, end))
        if end == duration:
            break
        start = end - overlap_seconds
        index += 1
    return chunks

def validate_pair(item):
    required = {"audio_path", "source_language", "target_language", "source_text", "target_text"}
    missing = required.difference(item)
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")
    if item["source_language"] == item["target_language"]:
        raise ValueError("source and target language must differ")
    return item

def latency_budget(asr_ms, mt_ms, tts_ms):
    values = [asr_ms, mt_ms, tts_ms]
    if any(value < 0 for value in values):
        raise ValueError("latency cannot be negative")
    return {"asr_ms": asr_ms, "mt_ms": mt_ms, "tts_ms": tts_ms, "total_ms": sum(values)}
