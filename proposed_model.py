"""
Proposed Method:
Structured Noise-Invariant Supervised Contrastive Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class ProposedIntentModel(nn.Module):
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

        # Classification head
        self.classifier = nn.Linear(hidden, num_labels)

        # Projection head (discarded at inference)
        self.projection = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, proj_dim),
        )

        self.temperature = temperature

    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state[:, 0]  # CLS

    def supervised_contrastive_loss(self, z, labels):
        z = F.normalize(z, dim=1)
        sim = torch.matmul(z, z.T) / self.temperature

        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float().to(z.device)

        logits_mask = torch.ones_like(mask) - torch.eye(
            mask.size(0), device=z.device
        )
        mask = mask * logits_mask

        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (
            mask.sum(dim=1) + 1e-12
        )

        loss = -mean_log_prob_pos.mean()
        return loss

    def forward(
        self,
        input_ids,
        attention_mask,
        noisy_input_ids=None,
        noisy_attention_mask=None,
        labels=None,
        lambda_cl=0.1,
    ):
        h_clean = self.encode(input_ids, attention_mask)
        logits = self.classifier(h_clean)

        loss = None
        if noisy_input_ids is not None and labels is not None:
            h_noisy = self.encode(
                noisy_input_ids, noisy_attention_mask
            )

            z_clean = self.projection(h_clean)
            z_noisy = self.projection(h_noisy)

            z_all = torch.cat([z_clean, z_noisy], dim=0)
            y_all = torch.cat([labels, labels], dim=0)

            loss_ce = F.cross_entropy(logits, labels)
            loss_cl = self.supervised_contrastive_loss(z_all, y_all)

            loss = loss_ce + lambda_cl * loss_cl

        return logits, loss
