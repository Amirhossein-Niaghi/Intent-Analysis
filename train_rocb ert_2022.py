import torch
from torch.optim import AdamW
from models.rocb ert_2022 import RoCBertText


def train_rocb ert(
    dataloader,
    tokenizer,
    num_labels,
    encoder_name="bert-base-multilingual-cased",
    epochs=5,
    lr=2e-5,
    device="cpu",
):
    model = RoCBertText(
        encoder_name=encoder_name,
        num_labels=num_labels,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):
        for batch in dataloader:
            clean_texts, noisy_texts, labels = batch

            clean_enc = tokenizer(
                clean_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            noisy_enc = tokenizer(
                noisy_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            labels = torch.tensor(labels).to(device)

            _, loss = model(
                clean_enc["input_ids"],
                clean_enc["attention_mask"],
                noisy_enc["input_ids"],
                noisy_enc["attention_mask"],
                labels,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[RoCBert] Epoch {epoch+1} | Loss: {loss.item():.4f}")

    return model
