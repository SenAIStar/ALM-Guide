import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--audio", required=True)
    args = parser.parse_args()
    cfg = json.load(open(args.config, encoding="utf-8"))
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model=cfg["output_dir"], chunk_length_s=cfg["max_audio_seconds"], return_timestamps=True)
    print(json.dumps(asr(args.audio, generate_kwargs={"language": cfg["language"], "task": cfg["task"]}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
