import torch
from torch.optim import AdamW
from models.proposed_model import ProposedIntentModel
from data.noise import apply_structured_noise


def train_proposed(
    dataloader,
    tokenizer,
    num_labels,
    encoder_name,
    alpha=0.4,
    lambda_cl=0.1,
    epochs=5,
    lr=2e-5,
    device="cpu",
):
    model = ProposedIntentModel(
        encoder_name=encoder_name,
        num_labels=num_labels,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        for texts, labels in dataloader:
            noisy_texts = [
                apply_structured_noise(t, alpha)
                for t in texts
            ]

            enc_clean = tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            enc_noisy = tokenizer(
                noisy_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            labels = torch.tensor(labels).to(device)

            _, loss = model(
                enc_clean["input_ids"],
                enc_clean["attention_mask"],
                enc_noisy["input_ids"],
                enc_noisy["attention_mask"],
                labels,
                lambda_cl=lambda_cl,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(
            f"[Proposed] Epoch {epoch+1} | Loss: {loss.item():.4f}"
        )

    return model
