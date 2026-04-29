"""
Adversarial Contrastive Learning (ACL) - 2025
Character-level adversarial robustness for text classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class ACLModel(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        num_labels: int,
        proj_dim: int = 128,
        temperature: float = 0.07,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size

        self.projection = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, proj_dim),
        )

        self.classifier = nn.Linear(hidden, num_labels)
        self.temperature = temperature

    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state[:, 0]

    def contrastive_loss(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        sim = torch.matmul(z1, z2.T) / self.temperature
        labels = torch.arange(z1.size(0)).to(z1.device)

        return F.cross_entropy(sim, labels)

    def forward(
        self,
        input_ids,
        attention_mask,
        adv_input_ids=None,
        adv_attention_mask=None,
        labels=None,
    ):
        cls = self.encode(input_ids, attention_mask)
        logits = self.classifier(cls)

        loss = None
        if adv_input_ids is not None and labels is not None:
            cls_adv = self.encode(adv_input_ids, adv_attention_mask)

            z = self.projection(cls)
            z_adv = self.projection(cls_adv)

            loss_ce = F.cross_entropy(logits, labels)
            loss_cl = self.contrastive_loss(z, z_adv)

            loss = loss_ce + loss_cl

        return logits, loss
