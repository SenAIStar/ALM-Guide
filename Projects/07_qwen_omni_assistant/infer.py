import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--audio")
    parser.add_argument("--image")
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", default="answer.wav")
    args = parser.parse_args()
    cfg = json.load(open(args.config, encoding="utf-8"))
    import soundfile as sf
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from qwen_omni_utils import process_mm_info
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(cfg["model_id"], torch_dtype="auto", device_map="auto")
    processor = Qwen2_5OmniProcessor.from_pretrained(cfg["model_id"])
    content = []
    if args.audio:
        content.append({"type": "audio", "audio": args.audio})
    if args.image:
        content.append({"type": "image", "image": args.image})
    content.append({"type": "text", "text": args.text})
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}],
        },
        {"role": "user", "content": content},
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=cfg["use_audio_in_video"])
    inputs = processor(text=prompt, audio=audios, images=images, videos=videos, padding=True, return_tensors="pt", use_audio_in_video=cfg["use_audio_in_video"])
    inputs = inputs.to(model.device).to(model.dtype)
    if cfg["enable_talker"]:
        text_ids, audio = model.generate(**inputs, use_audio_in_video=cfg["use_audio_in_video"], return_audio=True)
    else:
        model.disable_talker()
        text_ids = model.generate(**inputs, use_audio_in_video=cfg["use_audio_in_video"], return_audio=False)
        audio = None
    print(processor.batch_decode(text_ids, skip_special_tokens=True)[0])
    if audio is not None:
        sf.write(args.output, audio.reshape(-1).detach().cpu().numpy(), cfg["sample_rate"])

if __name__ == "__main__":
    main()
