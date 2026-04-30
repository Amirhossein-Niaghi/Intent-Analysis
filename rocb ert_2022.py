"""
RoCBert-style Robust Contrastive Training (Text-only)
ACL 2022 baseline adaptation for conversational intent classification.

Implements:
- Transformer encoder
- Dual-view contrastive objective
- Supervised classification head

Libraries:
- torch
- transformers
- numpy

No external AI/LLM frameworks used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class RoCBertText(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        num_labels: int,
        temperature: float = 0.1,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size

        self.classifier = nn.Linear(hidden_size, num_labels)
        self.temperature = temperature

    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_emb = outputs.last_hidden_state[:, 0]
        return F.normalize(cls_emb, dim=1)

    def contrastive_loss(self, z1, z2):
        """
        InfoNCE loss used in RoCBert textual branch
        """
        batch_size = z1.size(0)
        z = torch.cat([z1, z2], dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        labels = torch.arange(batch_size).to(z.device)
        labels = torch.cat([labels + batch_size, labels])

        mask = torch.eye(2 * batch_size).bool().to(z.device)
        sim = sim.masked_fill(mask, -1e9)

        loss = F.cross_entropy(sim, labels)
        return loss

    def forward(
        self,
        input_ids_clean,
        attention_mask_clean,
        input_ids_noisy=None,
        attention_mask_noisy=None,
        labels=None,
    ):
        z_clean = self.encode(input_ids_clean, attention_mask_clean)

        logits = self.classifier(z_clean)

        loss = None
        contrastive = None

        if input_ids_noisy is not None:
            z_noisy = self.encode(input_ids_noisy, attention_mask_noisy)
            contrastive = self.contrastive_loss(z_clean, z_noisy)

        if labels is not None:
            ce_loss = F.cross_entropy(logits, labels)
            loss = ce_loss
            if contrastive is not None:
                loss = loss + contrastive

        return logits, loss
