"""
Training script for Baseline Transformer (CE)
"""

import torch
from torch.optim import AdamW
from models.baseline_transformer import BaselineTransformer


def train_baseline(
    dataloader,
    tokenizer,
    num_labels,
    encoder_name,
    epochs=5,
    lr=2e-5,
    device="cpu",
):
    model = BaselineTransformer(
        encoder_name=encoder_name,
        num_labels=num_labels,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for texts, labels in dataloader:
            enc = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(device)

            labels = torch.tensor(labels).to(device)

            logits, loss = model(
                enc["input_ids"],
                enc["attention_mask"],
                labels,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(
            f"[Baseline CE] Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}"
        )

    return model
