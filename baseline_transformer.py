"""
Baseline Transformer for Intent Classification
Standard Fine-Tuning with Cross-Entropy Loss
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class BaselineTransformer(nn.Module):
    def __init__(self, encoder_name: str, num_labels: int):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size

        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cls_rep = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_rep)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return logits, loss
