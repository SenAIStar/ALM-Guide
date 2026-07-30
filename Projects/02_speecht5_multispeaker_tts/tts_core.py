from __future__ import annotations

import math


def normalize_embedding(values, expected_dim: int = 512):
    if len(values) != expected_dim:
        raise ValueError(f"expected {expected_dim} dimensions")
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        raise ValueError("zero speaker embedding")
    return [value / norm for value in values]


def validate_sample(item):
    required = {"audio_path", "text", "speaker_id", "speaker_embedding_path", "duration"}
    missing = required.difference(item)
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")
    if not item["text"].strip() or float(item["duration"]) <= 0:
        raise ValueError("invalid text or duration")
    return item


def cosine_similarity(a, b, eps=1e-12):
    if len(a) != len(b):
        raise ValueError("embedding dimensions differ")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / max(na * nb, eps)


def real_time_factor(audio_seconds, inference_seconds):
    if audio_seconds <= 0:
        raise ValueError("audio duration must be positive")
    return inference_seconds / audio_seconds
