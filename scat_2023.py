"""
SCAT: Self-supervised Contrastive Adversarial Training
Text-only implementation aligned with ACL 2023 formulation.

Implements:
- Transformer encoder
- Adversarial embedding perturbation
- Contrastive + cross-entropy loss

Libraries:
- torch
- transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class SCATModel(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        num_labels: int,
        epsilon: float = 1e-2,
        temperature: float = 0.1,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size

        self.classifier = nn.Linear(hidden, num_labels)
        self.epsilon = epsilon
        self.temperature = temperature

    def encode(self, input_ids, attention_mask, embed_noise=None):
        embeddings = self.encoder.embeddings.word_embeddings(input_ids)

        if embed_noise is not None:
            embeddings = embeddings + embed_noise

        outputs = self.encoder(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
        )

        cls = outputs.last_hidden_state[:, 0]
        return F.normalize(cls, dim=1)

    def contrastive_loss(self, z_clean, z_adv):
        batch_size = z_clean.size(0)
        z = torch.cat([z_clean, z_adv], dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        labels = torch.arange(batch_size).to(z.device)
        labels = torch.cat([labels + batch_size, labels])

        mask = torch.eye(2 * batch_size).bool().to(z.device)
        sim = sim.masked_fill(mask, -1e9)

        return F.cross_entropy(sim, labels)

    def forward(self, input_ids, attention_mask, labels=None):
        # Clean embedding
        z_clean = self.encode(input_ids, attention_mask)
        logits = self.classifier(z_clean)

        loss = None

        if labels is not None:
            ce_loss = F.cross_entropy(logits, labels)

            # Generate adversarial noise in embedding space
            z_clean_detached = z_clean.detach()
            z_clean_detached.requires_grad = True

            adv_logits = self.classifier(z_clean_detached)
            adv_loss = F.cross_entropy(adv_logits, labels)
            adv_loss.backward(retain_graph=True)

            grad = z_clean_detached.grad
            noise = self.epsilon * grad.sign()
            noise = noise.unsqueeze(1)

            # Adversarial embedding
            z_adv = self.encode(
                input_ids,
                attention_mask,
                embed_noise=noise,
            )

            contrastive = self.contrastive_loss(z_clean, z_adv)
            loss = ce_loss + contrastive

        return logits, loss
