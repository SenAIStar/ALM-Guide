from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ManifestRow:
    audio_path: str
    text: str
    language: str
    speaker_id: str
    duration: float
    split: str

    @classmethod
    def from_dict(cls, value):
        row = cls(**{key: value[key] for key in cls.__annotations__})
        if not row.audio_path or not row.text.strip():
            raise ValueError("audio_path and text are required")
        if row.duration <= 0:
            raise ValueError("duration must be positive")
        if row.split not in {"train", "validation", "test"}:
            raise ValueError("invalid split")
        return row


def normalize_text(text: str, keep_punctuation: bool = False) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    if not keep_punctuation:
        text = re.sub(r"[^0-9a-z一-鿿 ]", "", text)
    return text


def assert_speaker_disjoint(rows) -> None:
    owners = {}
    for row in rows:
        previous = owners.setdefault(row.speaker_id, row.split)
        if previous != row.split:
            raise ValueError(f"speaker leakage: {row.speaker_id}")


def edit_distance(reference, hypothesis) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, start=1):
        current = [i]
        for j, hyp_token in enumerate(hypothesis, start=1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (ref_token != hyp_token),
            ))
        previous = current
    return previous[-1]


def error_rate(reference: str, hypothesis: str, unit: str = "word") -> float:
    ref = list(reference) if unit == "char" else reference.split()
    hyp = list(hypothesis) if unit == "char" else hypothesis.split()
    if not ref:
        raise ValueError("empty reference")
    return edit_distance(ref, hyp) / len(ref)
