class WhisperCollator:
    def __init__(self, processor, decoder_start_token_id=None):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, features):
        label_features = [{"input_ids": item["labels"]} for item in features]
        input_features = [{"input_features": item["input_features"]} for item in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        label_ids = labels["input_ids"].masked_fill(labels["attention_mask"].ne(1), -100)
        if self.decoder_start_token_id is not None and (label_ids[:, 0] == self.decoder_start_token_id).all():
            label_ids = label_ids[:, 1:]
        batch["labels"] = label_ids
        return batch
