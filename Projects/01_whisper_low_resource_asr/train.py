import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--use-lora", action="store_true")
    args = parser.parse_args()
    cfg = json.load(open(args.config, encoding="utf-8"))

    from datasets import load_dataset
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
    from collator import WhisperCollator

    processor = AutoProcessor.from_pretrained(cfg["model_id"], language=cfg["language"], task=cfg["task"])
    model = AutoModelForSpeechSeq2Seq.from_pretrained(cfg["model_id"])
    if args.use_lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05))

    dataset = load_dataset("json", data_files={"train": cfg["train_manifest"], "validation": cfg["validation_manifest"]})

    def encode(item):
        import soundfile as sf
        import torch
        import torchaudio
        audio, sample_rate = sf.read(item["audio_path"], always_2d=True)
        waveform = torch.tensor(audio.mean(axis=1), dtype=torch.float32)
        target_sr = int(cfg["sample_rate"])
        if sample_rate != target_sr:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
        item["input_features"] = processor.feature_extractor(waveform.numpy(), sampling_rate=target_sr).input_features[0]
        item["labels"] = processor.tokenizer(item["text"]).input_ids
        return item

    dataset = dataset.map(encode, remove_columns=dataset["train"].column_names)
    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg["output_dir"], learning_rate=1e-5, per_device_train_batch_size=8,
        gradient_accumulation_steps=2, eval_strategy="steps", save_steps=500,
        eval_steps=500, fp16=bool(cfg.get("fp16", True)), predict_with_generate=True, report_to=["tensorboard"],
    )
    trainer = Seq2SeqTrainer(
        model=model, args=training_args, train_dataset=dataset["train"], eval_dataset=dataset["validation"],
        data_collator=WhisperCollator(processor, model.config.decoder_start_token_id), processing_class=processor,
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    processor.save_pretrained(cfg["output_dir"])


if __name__ == "__main__":
    main()
