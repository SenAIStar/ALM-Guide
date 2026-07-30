import argparse
import json

from grounding_core import best_span, window_starts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--features", help="Question-conditioned [frames, hidden] tensor")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    cfg = json.load(open(args.config, encoding="utf-8"))
    import soundfile as sf
    info = sf.info(args.audio)
    duration = info.frames / info.samplerate
    candidates = window_starts(duration, cfg["chunk_seconds"], cfg["overlap_seconds"])
    if not args.features and not args.checkpoint:
        print(json.dumps({"question": args.question, "duration": duration, "candidate_window_starts": candidates, "status": "windows_prepared"}, ensure_ascii=False, indent=2))
        return
    if not args.features or not args.checkpoint:
        raise ValueError("--features and --checkpoint must be provided together")

    import torch
    from modeling import build_model
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = build_model(saved["hidden_size"], saved["num_speakers"], saved["num_answers"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    features = torch.load(args.features, map_location="cpu", weights_only=True).float()
    if features.ndim != 2:
        raise ValueError("features must have shape [frames, hidden_size]")
    mask = torch.ones((1, features.shape[0]), dtype=torch.long)
    with torch.inference_mode():
        output = model(features.unsqueeze(0), mask)
    start_index, end_index = best_span(
        output["start_logits"][0].tolist(),
        output["end_logits"][0].tolist(),
    )
    print(json.dumps({
        "question": args.question,
        "start_index": start_index,
        "end_index": end_index,
        "speaker_index": int(output["speaker_logits"].argmax(-1)[0]),
        "answer_index": int(output["answer_logits"].argmax(-1)[0]),
        "status": "head_inference_complete",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
