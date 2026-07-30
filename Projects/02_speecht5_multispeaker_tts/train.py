import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    cfg = json.load(open(args.config, encoding="utf-8"))
    from datasets import load_dataset
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, SpeechT5ForTextToSpeech, SpeechT5Processor
    from collator import SpeechT5Collator
    processor = SpeechT5Processor.from_pretrained(cfg["model_id"])
    model = SpeechT5ForTextToSpeech.from_pretrained(cfg["model_id"])
    model.config.use_cache = False
    dataset = load_dataset("json", data_files={"train": cfg["manifest"]})["train"]

    def encode(item):
        import json as json_module
        import soundfile as sf
        import torch
        import torchaudio
        from tts_core import normalize_embedding
        audio, sample_rate = sf.read(item["audio_path"], always_2d=True)
        waveform = torch.tensor(audio.mean(axis=1), dtype=torch.float32)
        target_sr = int(cfg["sample_rate"])
        if sample_rate != target_sr:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
        encoded = processor(text=item["text"], audio_target=waveform.numpy(), sampling_rate=target_sr)
        encoded["labels"] = encoded["labels"][0]
        embedding = normalize_embedding(json_module.load(open(item["speaker_embedding_path"], encoding="utf-8")))
        if len(embedding) != int(model.config.speaker_embedding_dim):
            raise ValueError(f"speaker embedding must have {model.config.speaker_embedding_dim} values")
        encoded["speaker_embeddings"] = embedding
        return encoded

    dataset = dataset.map(encode, remove_columns=dataset.column_names)
    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg["output_dir"], per_device_train_batch_size=4, gradient_accumulation_steps=4,
        learning_rate=1e-5, save_steps=500, logging_steps=25, gradient_checkpointing=True, report_to=["tensorboard"],
    )
    trainer = Seq2SeqTrainer(model=model, args=training_args, train_dataset=dataset, data_collator=SpeechT5Collator(processor, reduction_factor=model.config.reduction_factor), processing_class=processor)
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    processor.save_pretrained(cfg["output_dir"])


if __name__ == "__main__":
    main()
