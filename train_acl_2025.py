import torch
from torch.optim import AdamW
from models.acl_2022 import ACLModel
from data.adversarial_noise import adversarial_char_noise


def train_acl(
    dataloader,
    tokenizer,
    num_labels,
    encoder_name="bert-base-multilingual-cased",
    epochs=5,
    lr=2e-5,
    device="cpu",
):
    model = ACLModel(
        encoder_name=encoder_name,
        num_labels=num_labels,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        for texts, labels in dataloader:
            adv_texts = [
                adversarial_char_noise(t) for t in texts
            ]

            enc = tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            adv_enc = tokenizer(
                adv_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            labels = torch.tensor(labels).to(device)

            _, loss = model(
                enc["input_ids"],
                enc["attention_mask"],
                adv_enc["input_ids"],
                adv_enc["attention_mask"],
                labels,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[ACL] Epoch {epoch+1} | Loss: {loss.item():.4f}")

    return model
