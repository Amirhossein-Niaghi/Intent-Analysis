"""
Data Augmentation Baseline
--------------------------
Standard Transformer fine-tuning with label-preserving noisy samples
added to the training data. Uses cross-entropy loss only.

This corresponds to Section 5.3.2 of the paper.
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, AdamW


class AugmentedIntentModel(nn.Module):
    def __init__(self, encoder_name, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_embedding = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_embedding)
        return logits


def train_data_augmentation_baseline(
    model,
    tokenizer,
    train_texts,
    train_labels,
    noise_fn,
    alpha_train=0.3,
    batch_size=16,
    epochs=5,
    lr=2e-5,
    device="cpu",
    seed=42,
):
    """
    Train baseline with data augmentation.
    """

    torch.manual_seed(seed)
    random.seed(seed)

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(epochs):
        indices = list(range(len(train_texts)))
        random.shuffle(indices)

        total_loss = 0.0

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i : i + batch_size]

            clean_texts = [train_texts[j] for j in batch_idx]
            labels = torch.tensor(
                [train_labels[j] for j in batch_idx],
                device=device,
            )

            # Generate noisy versions
            noisy_texts = [
                noise_fn(text, alpha_train) for text in clean_texts
            ]

            # Combine clean + noisy
            all_texts = clean_texts + noisy_texts
            all_labels = torch.cat([labels, labels], dim=0)

            encoded = tokenizer(
                all_texts,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(device)

            logits = model(
                encoded["input_ids"],
                encoded["attention_mask"],
            )

            loss = criterion(logits, all_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (len(indices) // batch_size + 1)
        print(
            f"[Epoch {epoch+1}/{epochs}] "
            f"Augmentation CE Loss: {avg_loss:.4f}"
        )

    return model
