from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate_row(row):
    required = {"feature_path", "start_index", "end_index", "speaker_index", "answer_index"}
    missing = required.difference(row)
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")
    if row["start_index"] < 0 or row["end_index"] < row["start_index"]:
        raise ValueError("invalid target span")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="outputs/heads.pt")
    args = parser.parse_args()
    cfg = json.load(open(args.config, encoding="utf-8"))

    import torch
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import DataLoader, Dataset
    from modeling import build_model

    rows = [validate_row(json.loads(line)) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]

    class FeatureDataset(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            row = rows[index]
            features = torch.load(row["feature_path"], map_location="cpu", weights_only=True).float()
            if features.ndim != 2:
                raise ValueError("feature tensor must have shape [frames, hidden_size]")
            if row["end_index"] >= features.shape[0]:
                raise ValueError("target span exceeds feature length")
            return features, row

    def collate(items):
        tensors = [item[0] for item in items]
        features = pad_sequence(tensors, batch_first=True)
        mask = torch.zeros(features.shape[:2], dtype=torch.long)
        for index, tensor in enumerate(tensors):
            mask[index, :tensor.shape[0]] = 1
        values = {key: torch.tensor([item[1][key] for item in items], dtype=torch.long) for key in ("start_index", "end_index", "speaker_index", "answer_index")}
        return {"features": features, "attention_mask": mask, **values}

    dataset = FeatureDataset()
    first_features, _ = dataset[0]
    model = build_model(first_features.shape[1], int(cfg.get("num_speakers", 64)), int(cfg.get("num_answers", 512)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 3e-4)))
    loader = DataLoader(dataset, batch_size=int(cfg.get("batch_size", 8)), shuffle=True, collate_fn=collate)
    model.train()
    for epoch in range(int(cfg.get("epochs", 10))):
        total = 0.0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            output["loss"].backward()
            optimizer.step()
            total += float(output["loss"].detach())
        print(json.dumps({"epoch": epoch + 1, "mean_loss": total / max(len(loader), 1)}))
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "hidden_size": first_features.shape[1], "num_speakers": int(cfg.get("num_speakers", 64)), "num_answers": int(cfg.get("num_answers", 512))}, target)


if __name__ == "__main__":
    main()
