from __future__ import annotations
from collections import deque
from dataclasses import dataclass

@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    score: float = 1.0
    speaker: str = "unknown"

    def __post_init__(self):
        if self.start < 0 or self.end <= self.start:
            raise ValueError("invalid interval")

def temporal_iou(a: Interval, b: Interval) -> float:
    intersection = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return intersection / union

def temporal_nms(items, threshold: float):
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    kept = []
    for item in sorted(items, key=lambda x: x.score, reverse=True):
        if all(item.speaker != existing.speaker or temporal_iou(item, existing) < threshold for existing in kept):
            kept.append(item)
    return kept

def best_span(start_scores, end_scores, max_span=None):
    if len(start_scores) != len(end_scores) or not start_scores:
        raise ValueError("start and end scores must be non-empty and have equal length")
    if max_span is not None and max_span <= 0:
        raise ValueError("max_span must be positive")
    candidates = deque()
    best_start, best_end, best_score = 0, 0, float("-inf")
    for end, end_score in enumerate(end_scores):
        while candidates and float(start_scores[candidates[-1]]) <= float(start_scores[end]):
            candidates.pop()
        candidates.append(end)
        minimum_start = 0 if max_span is None else end - max_span + 1
        while candidates[0] < minimum_start:
            candidates.popleft()
        start = candidates[0]
        score = float(start_scores[start]) + float(end_score)
        if score > best_score:
            best_start, best_end, best_score = start, end, score
    return best_start, best_end

def grounded_correct(prediction, reference, answer_correct: bool, threshold=0.5):
    return bool(answer_correct and prediction.speaker == reference.speaker and temporal_iou(prediction, reference) >= threshold)

def window_starts(duration, window, overlap):
    if duration <= 0 or window <= 0:
        raise ValueError("duration and window must be positive")
    if not 0 <= overlap < window:
        raise ValueError("invalid overlap")
    starts, step, current = [], window - overlap, 0.0
    while current < duration:
        starts.append(current)
        if current + window >= duration:
            break
        current += step
    return starts
