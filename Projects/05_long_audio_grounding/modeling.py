def build_model(hidden_size: int, num_speakers: int, num_answers: int):
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    class GroundingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.frame_score = nn.Linear(hidden_size, 2)
            self.speaker_head = nn.Linear(hidden_size, num_speakers)
            self.answer_head = nn.Linear(hidden_size, num_answers)

        def forward(self, features, attention_mask, start_index=None, end_index=None, speaker_index=None, answer_index=None):
            frame_logits = self.frame_score(features)
            invalid = attention_mask.eq(0).unsqueeze(-1)
            frame_logits = frame_logits.masked_fill(invalid, torch.finfo(frame_logits.dtype).min)
            weights = attention_mask.to(features.dtype).unsqueeze(-1)
            pooled = (features * weights).sum(1) / weights.sum(1).clamp_min(1.0)
            output = {
                "start_logits": frame_logits[..., 0],
                "end_logits": frame_logits[..., 1],
                "speaker_logits": self.speaker_head(pooled),
                "answer_logits": self.answer_head(pooled),
            }
            if start_index is not None:
                losses = {
                    "start": functional.cross_entropy(output["start_logits"], start_index),
                    "end": functional.cross_entropy(output["end_logits"], end_index),
                    "speaker": functional.cross_entropy(output["speaker_logits"], speaker_index),
                    "answer": functional.cross_entropy(output["answer_logits"], answer_index),
                }
                output["loss"] = sum(losses.values())
                output["losses"] = {key: value.detach() for key, value in losses.items()}
            return output

    return GroundingModel()
