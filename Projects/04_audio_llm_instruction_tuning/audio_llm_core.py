from __future__ import annotations

ALLOWED_TASKS = {"asr", "audio_event", "speaker", "emotion", "paralinguistic", "audio_qa"}

def validate_instruction_sample(item):
    required = {"sample_id", "audio_path", "instruction", "answer", "task"}
    missing = required.difference(item)
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")
    if item["task"] not in ALLOWED_TASKS:
        raise ValueError(f"unsupported task: {item['task']}")
    if not item["instruction"].strip() or not item["answer"].strip():
        raise ValueError("instruction and answer cannot be empty")
    return item

def task_macro_average(records, score_key="score"):
    buckets = {}
    for row in records:
        buckets.setdefault(row["task"], []).append(float(row[score_key]))
    if not buckets:
        raise ValueError("no records")
    per_task = {task: sum(values) / len(values) for task, values in buckets.items()}
    return {"per_task": per_task, "macro": sum(per_task.values()) / len(per_task)}

def downsample_mask(mask, output_frames):
    if output_frames <= 0:
        raise ValueError("output_frames must be positive")
    if not mask:
        return [0] * output_frames
    result = []
    for index in range(output_frames):
        left = index * len(mask) // output_frames
        right = max((index + 1) * len(mask) // output_frames, left + 1)
        result.append(int(any(mask[left:right])))
    return result
