from __future__ import annotations

ALLOWED = {"text", "image", "video", "audio"}

def validate_sample(item):
    required = {"sample_id", "modalities", "instruction", "answer", "conflict_type"}
    missing = required.difference(item)
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")
    modalities = set(item["modalities"])
    if not modalities or not modalities.issubset(ALLOWED):
        raise ValueError("unsupported modalities")
    if item["conflict_type"] != "none" and len(modalities) < 2:
        raise ValueError("conflict sample requires at least two modalities")
    return item

def counterfactual_report(full_score, ablated_scores, required_modalities):
    report = {"full": float(full_score)}
    for modality, score in ablated_scores.items():
        report[f"drop_without_{modality}"] = float(full_score) - float(score)
        report[f"expected_dependency_{modality}"] = modality in required_modalities
    return report

def conflict_preference(predicted_sources):
    counts = {"audio": 0, "vision": 0, "text": 0, "other": 0}
    for source in predicted_sources:
        counts[source if source in counts else "other"] += 1
    return counts
