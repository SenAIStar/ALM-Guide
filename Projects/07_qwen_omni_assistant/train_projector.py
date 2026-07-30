from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--input-size", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--output", default="outputs/projector.pt")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader, Dataset

    rows = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("manifest must contain at least one sample")
    required = {"source_feature_path", "target_feature_path", "modality"}
    for row in rows:
        missing = required.difference(row)
        if missing:
            raise ValueError(f"missing fields: {sorted(missing)}")

    class PairDataset(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            row = rows[index]
            source = torch.load(row["source_feature_path"], map_location="cpu", weights_only=True).float().mean(0)
            target = torch.load(row["target_feature_path"], map_location="cpu", weights_only=True).float().mean(0)
            if source.numel() != args.input_size or target.numel() != args.hidden_size:
                raise ValueError("feature dimension does not match CLI sizes")
            return source, target

    projector = nn.Sequential(nn.LayerNorm(args.input_size), nn.Linear(args.input_size, args.hidden_size), nn.GELU(), nn.Linear(args.hidden_size, args.hidden_size))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    projector.to(device)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=3e-4)
    loader = DataLoader(PairDataset(), batch_size=16, shuffle=True)
    for epoch in range(args.epochs):
        total = 0.0
        for source, target in loader:
            source, target = source.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = projector(source)
            loss = 1.0 - functional.cosine_similarity(prediction, target, dim=-1).mean()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        print(json.dumps({"epoch": epoch + 1, "mean_loss": total / max(len(loader), 1)}))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": projector.state_dict(), "input_size": args.input_size, "hidden_size": args.hidden_size}, output)


if __name__ == "__main__":
    main()
