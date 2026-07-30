from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_rows(path):
    from s2st_core import validate_pair
    return [validate_pair(json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validation-manifest")
    args = parser.parse_args()
    cfg = json.load(open(args.config, encoding="utf-8"))

    import soundfile as sf
    import torch
    import torchaudio
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import Dataset
    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText, Trainer, TrainingArguments

    processor = AutoProcessor.from_pretrained(cfg["model_id"])
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(cfg["model_id"])
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=int(cfg.get("lora_r", 8)),
        lora_alpha=int(cfg.get("lora_alpha", 16)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        target_modules=cfg.get("lora_target_modules", ["q_proj", "v_proj"]),
    ))
    model.print_trainable_parameters()

    class TranslationDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            audio, sample_rate = sf.read(row["audio_path"], always_2d=True)
            waveform = torch.tensor(audio.mean(axis=1), dtype=torch.float32)
            target_sr = int(cfg["sample_rate"])
            if sample_rate != target_sr:
                waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
            features = processor(audios=waveform.numpy(), sampling_rate=target_sr, return_tensors="pt")
            attention_mask = features.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones(features["input_features"].shape[:2], dtype=torch.long)
            labels = processor.tokenizer(
                row["target_text"], src_lang=row["target_language"], return_tensors="pt"
            )["input_ids"][0]
            return {
                "input_features": features["input_features"][0],
                "attention_mask": attention_mask[0],
                "labels": labels,
            }

    def collate(items):
        feature_batch = processor.feature_extractor.pad(
            [
                {
                    "input_features": item["input_features"],
                    "attention_mask": item["attention_mask"],
                }
                for item in items
            ],
            return_tensors="pt",
        )
        label_batch = processor.tokenizer.pad(
            [{"input_ids": item["labels"]} for item in items], return_tensors="pt"
        )
        feature_batch["labels"] = label_batch["input_ids"].masked_fill(label_batch["attention_mask"].ne(1), -100)
        return feature_batch

    train_dataset = TranslationDataset(read_rows(args.manifest))
    eval_dataset = TranslationDataset(read_rows(args.validation_manifest)) if args.validation_manifest else None
    training_args = TrainingArguments(
        output_dir=cfg.get("output_dir", "outputs/seamless-s2tt-lora"),
        learning_rate=float(cfg.get("learning_rate", 1e-4)),
        per_device_train_batch_size=int(cfg.get("batch_size", 2)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
        num_train_epochs=float(cfg.get("epochs", 3)),
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=int(cfg.get("eval_steps", 250)),
        save_steps=int(cfg.get("save_steps", 250)),
        logging_steps=int(cfg.get("logging_steps", 20)),
        bf16=bool(cfg.get("bf16", False)),
        fp16=bool(cfg.get("fp16", True)),
        remove_unused_columns=False,
        report_to=["tensorboard"],
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=collate)
    trainer.train()
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
