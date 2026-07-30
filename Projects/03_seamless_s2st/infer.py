import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = json.load(open(args.config, encoding="utf-8"))
    import soundfile as sf
    import torch
    import torchaudio
    from transformers import AutoProcessor, SeamlessM4Tv2Model
    processor = AutoProcessor.from_pretrained(cfg["model_id"])
    model = SeamlessM4Tv2Model.from_pretrained(cfg["model_id"]).eval()
    audio, sample_rate = sf.read(args.audio, always_2d=True)
    waveform = torch.tensor(audio.mean(axis=1), dtype=torch.float32)
    target_sr = int(cfg["sample_rate"])
    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
    inputs = processor(audios=waveform.numpy(), sampling_rate=target_sr, return_tensors="pt")
    with torch.inference_mode():
        generated = model.generate(**inputs, tgt_lang=cfg["target_language"], generate_speech=True)
    waveform = generated[0].cpu().float().numpy().squeeze()
    sf.write(args.output, waveform, model.config.sampling_rate)

if __name__ == "__main__":
    main()
