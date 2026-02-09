import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from practice_CE_3_dataset_dataloader import SpeechDataset
from practice_CE_4_hubert_bilstm import HubertCTC
from practice_CE_collate_fn import collate_fn
from tqdm import tqdm 


MODEL_NAME = "yky-h/japanese-hubert-base"
EPOCHS = 20

TRAIN_MANIFEST = "data/train_manifest_2.json"
VALID_MANIFEST = "data/valid_manifest.json"


# =====================
# Dataset
# =====================
train_dataset = SpeechDataset(
    TRAIN_MANIFEST,
    MODEL_NAME
)

valid_dataset = SpeechDataset(
    VALID_MANIFEST,
    MODEL_NAME
)

NUM_CLASSES = train_dataset.num_classes  # ← train 基準で固定！


train_loader = DataLoader(
    train_dataset,
    batch_size=20,
    shuffle=True,
    collate_fn=collate_fn
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_fn
)


# =====================
# Model
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HubertCTC(MODEL_NAME, NUM_CLASSES).to(device)


# =====================
# Loss & Optimizer
# =====================
# =====================
# Class weights
# =====================
counts = torch.tensor(train_dataset.class_counts, dtype=torch.float)

weights = 1.0 / counts

# print(weights)


weights = weights.to(device)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-4)


best_valid_loss = float("inf")


# =====================
# Train & Validation loop
# =====================
for epoch in range(EPOCHS):

    # ---------- Train ----------
    model.train()
    train_loss = 0.0

    train_pbar = tqdm(
        train_loader,
        desc=f"[Train] Epoch {epoch}",
        leave=False
    )

    for wavs, wav_lens, labels in train_pbar:
        wavs = wavs.to(device)
        wav_lens = wav_lens.to(device)
        labels = labels.to(device)

        logits = model(wavs, wav_lens)   # [B, num_classes]
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_pbar.set_postfix(loss=f"{loss.item():.4f}")

    train_loss /= len(train_loader)


    # ---------- Validation ----------
    model.eval()
    valid_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        valid_pbar = tqdm(
            valid_loader,
            desc=f"[Valid] Epoch {epoch}",
            leave=False
        )

        for wavs, wav_lens, labels in valid_pbar:
            wavs = wavs.to(device)
            wav_lens = wav_lens.to(device)
            labels = labels.to(device)

            logits = model(wavs, wav_lens)
            loss = criterion(logits, labels)

            valid_loss += loss.item()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    valid_loss /= len(valid_loader)
    valid_acc = correct / total if total > 0 else 0.0


    # ---------- Log ----------
    print(
        f"Epoch {epoch} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Valid Loss: {valid_loss:.4f} | "
        f"Valid Acc: {valid_acc:.4f}"
    )


    # ---------- Save best ----------
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "best_model.pt")
        print("★ Best model updated")


print("Training finished")
