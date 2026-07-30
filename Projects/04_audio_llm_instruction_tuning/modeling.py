def build_connector(input_size: int, hidden_size: int):
    import torch.nn as nn
    return nn.Sequential(
        nn.LayerNorm(input_size),
        nn.Linear(input_size, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, hidden_size),
    )


class AudioPrefixLM:
    @staticmethod
    def build(audio_encoder, connector, llm):
        import torch
        import torch.nn as nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.audio_encoder = audio_encoder
                self.connector = connector
                self.llm = llm

            def forward(self, input_values, audio_attention_mask, input_ids, attention_mask, labels):
                outputs = self.audio_encoder(input_values=input_values, attention_mask=audio_attention_mask)
                hidden = outputs.last_hidden_state
                if hasattr(self.audio_encoder, "_get_feature_vector_attention_mask"):
                    frame_mask = self.audio_encoder._get_feature_vector_attention_mask(hidden.shape[1], audio_attention_mask)
                    weights = frame_mask.to(hidden.dtype).unsqueeze(-1)
                    pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1.0)
                else:
                    pooled = hidden.mean(1)
                audio_token = self.connector(pooled).unsqueeze(1)
                text_embeddings = self.llm.get_input_embeddings()(input_ids)
                inputs_embeds = torch.cat([audio_token, text_embeddings], dim=1)
                prefix_mask = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
                full_mask = torch.cat([prefix_mask, attention_mask], dim=1)
                prefix_labels = torch.full((labels.shape[0], 1), -100, dtype=labels.dtype, device=labels.device)
                full_labels = torch.cat([prefix_labels, labels], dim=1)
                return self.llm(inputs_embeds=inputs_embeds, attention_mask=full_mask, labels=full_labels)

        return Model()
