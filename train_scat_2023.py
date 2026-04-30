import torch
from torch.optim import AdamW
from models.scat_2023 import SCATModel


def train_scat(
    dataloader,
    tokenizer,
    num_labels,
    encoder_name="bert-base-multilingual-cased",
    epochs=5,
    lr=2e-5,
    device="cpu",
):
    model = SCATModel(
        encoder_name=encoder_name,
        num_labels=num_labels,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        for texts, labels in dataloader:
            enc = tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            labels = torch.tensor(labels).to(device)

            _, loss = model(
                enc["input_ids"],
                enc["attention_mask"],
                labels,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[SCAT] Epoch {epoch+1} | Loss: {loss.item():.4f}")

    return model
