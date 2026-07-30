import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--text", required=True)
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = json.load(open(args.config, encoding="utf-8"))
    import soundfile as sf
    import torch
    from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor
    from tts_core import normalize_embedding
    processor = SpeechT5Processor.from_pretrained(cfg["model_id"])
    model = SpeechT5ForTextToSpeech.from_pretrained(cfg["output_dir"]).eval()
    vocoder = SpeechT5HifiGan.from_pretrained(cfg["vocoder_id"]).eval()
    speaker = torch.tensor(normalize_embedding(json.load(open(args.speaker, encoding="utf-8"))), dtype=torch.float32).unsqueeze(0)
    inputs = processor(text=args.text, return_tensors="pt")
    with torch.inference_mode():
        wav = model.generate_speech(inputs["input_ids"], speaker, vocoder=vocoder)
    sf.write(args.output, wav.cpu().numpy(), cfg["sample_rate"])

if __name__ == "__main__":
    main()
