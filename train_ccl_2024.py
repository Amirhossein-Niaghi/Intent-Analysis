import torch
from torch.optim import AdamW
from models.ccl_2024 import CCLModel
from data.counterfactuals import generate_counterfactual


def train_ccl(
    dataloader,
    tokenizer,
    num_labels,
    encoder_name="bert-base-multilingual-cased",
    epochs=5,
    lr=2e-5,
    device="cpu",
):
    model = CCLModel(
        encoder_name=encoder_name,
        num_labels=num_labels,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        for texts, labels in dataloader:
            cf_texts = [generate_counterfactual(t) for t in texts]

            enc = tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            enc_cf = tokenizer(
                cf_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            labels = torch.tensor(labels).to(device)

            _, loss = model(
                enc["input_ids"],
                enc["attention_mask"],
                labels,
                enc_cf["input_ids"],
                enc_cf["attention_mask"],
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[CCL] Epoch {epoch+1} | Loss: {loss.item():.4f}")

    return model
