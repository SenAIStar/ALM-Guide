from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    cfg = json.load(open(args.config, encoding="utf-8"))

    import soundfile as sf
    import torch
    import torchaudio
    from torch.utils.data import Dataset
    from transformers import AutoFeatureExtractor, AutoModel, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from audio_llm_core import validate_instruction_sample
    from modeling import AudioPrefixLM, build_connector

    rows = [validate_instruction_sample(json.loads(line)) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    feature_extractor = AutoFeatureExtractor.from_pretrained(cfg["audio_encoder_id"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["llm_id"], use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    audio_encoder = AutoModel.from_pretrained(cfg["audio_encoder_id"])
    llm = AutoModelForCausalLM.from_pretrained(cfg["llm_id"])
    if cfg.get("freeze_audio_encoder", True):
        audio_encoder.requires_grad_(False)
    if cfg.get("use_lora", True):
        from peft import LoraConfig, get_peft_model
        llm = get_peft_model(llm, LoraConfig(
            r=int(cfg.get("lora_r", 8)), lora_alpha=int(cfg.get("lora_alpha", 16)),
            lora_dropout=float(cfg.get("lora_dropout", 0.05)),
            target_modules=cfg.get("lora_target_modules", ["q_proj", "v_proj"]), task_type="CAUSAL_LM",
        ))
    connector = build_connector(audio_encoder.config.hidden_size, llm.config.hidden_size)
    model = AudioPrefixLM.build(audio_encoder, connector, llm)

    class InstructionDataset(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            return rows[index]

    def collate(items):
        arrays, prefixes, full_texts = [], [], []
        target_sr = int(cfg["sample_rate"])
        for item in items:
            audio, sample_rate = sf.read(item["audio_path"], always_2d=True)
            waveform = torch.tensor(audio.mean(axis=1), dtype=torch.float32)
            if sample_rate != target_sr:
                waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
            arrays.append(waveform.numpy())
            prefix = f"用户：{item['instruction']}\n助手："
            prefixes.append(prefix)
            full_texts.append(prefix + item["answer"] + tokenizer.eos_token)
        audio_batch = feature_extractor(arrays, sampling_rate=target_sr, padding=True, return_tensors="pt", return_attention_mask=True)
        text_batch = tokenizer(full_texts, padding=True, truncation=True, max_length=int(cfg.get("max_text_tokens", 512)), return_tensors="pt")
        labels = text_batch["input_ids"].clone()
        labels[text_batch["attention_mask"].eq(0)] = -100
        for row_index, prefix in enumerate(prefixes):
            prefix_length = len(tokenizer(prefix, add_special_tokens=True)["input_ids"])
            labels[row_index, :prefix_length] = -100
        return {
            "input_values": audio_batch["input_values"],
            "audio_attention_mask": audio_batch["attention_mask"],
            "input_ids": text_batch["input_ids"],
            "attention_mask": text_batch["attention_mask"],
            "labels": labels,
        }

    training_args = TrainingArguments(
        output_dir=cfg.get("output_dir", "outputs/audio-llm-adapter"),
        learning_rate=float(cfg.get("learning_rate", 1e-4)),
        per_device_train_batch_size=int(cfg.get("batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 16)),
        num_train_epochs=float(cfg.get("epochs", 3)),
        save_steps=int(cfg.get("save_steps", 250)),
        logging_steps=int(cfg.get("logging_steps", 10)),
        bf16=bool(cfg.get("bf16", False)),
        fp16=bool(cfg.get("fp16", True)),
        remove_unused_columns=False,
        report_to=["tensorboard"],
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=InstructionDataset(), data_collator=collate)
    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
