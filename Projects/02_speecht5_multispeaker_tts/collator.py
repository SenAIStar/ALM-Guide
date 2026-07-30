class SpeechT5Collator:
    def __init__(self, processor, speaker_dim=512, reduction_factor=2):
        self.processor = processor
        self.speaker_dim = speaker_dim
        self.reduction_factor = reduction_factor

    def __call__(self, features):
        import torch
        text = [{"input_ids": item["input_ids"]} for item in features]
        audio = [{"input_values": item["labels"]} for item in features]
        batch = self.processor.pad(input_ids=text, labels=audio, return_tensors="pt")
        batch["labels"] = batch["labels"].masked_fill(batch["decoder_attention_mask"].ne(1).unsqueeze(-1), -100)
        batch.pop("decoder_attention_mask")
        if self.reduction_factor > 1:
            lengths = torch.tensor([len(item["input_values"]) for item in audio])
            lengths = lengths - lengths.remainder(self.reduction_factor)
            batch["labels"] = batch["labels"][:, :int(lengths.max())]
        embeddings = [item["speaker_embeddings"] for item in features]
        batch["speaker_embeddings"] = torch.tensor(embeddings, dtype=torch.float32)
        return batch
