import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from practice_CE_3_dataset_dataloader import SpeechDataset
from practice_CE_4_hubert_bilstm import HubertCTC
from practice_CE_collate_fn import collate_fn

MODEL_NAME = "yky-h/japanese-hubert-base"
TEST_MANIFEST = "data/test_manifest_2.json"
TRAIN_MANIFEST = "data/train_manifest_2.json"
MODEL_PATH = "best_model.pt"
BATCH_SIZE = 1   # 出力を分かりやすくするため 1 推奨

# =====================
# Dataset / Loader
# =====================
train_dataset = SpeechDataset(TRAIN_MANIFEST, MODEL_NAME)
test_dataset = SpeechDataset(TEST_MANIFEST, MODEL_NAME)


loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)
# =====================
# Model
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = HubertCTC(
    MODEL_NAME,
    train_dataset.num_classes
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# =====================
# 推論ループ
# =====================
un_id = train_dataset.filler2id["うん"]
LOGIT_PENALTY = 0.03

correct = 0
total = 0

print("idx | GT | Pred | Prob")
print("-" * 40)

with torch.no_grad():
    for idx, (wavs, wav_lens, labels) in enumerate(tqdm(loader)):
        wavs = wavs.to(device)
        wav_lens = wav_lens.to(device)
        labels = labels.to(device)

        logits = model(wavs, wav_lens)     # [1, C]
        logits[:, un_id] -= LOGIT_PENALTY
        a_id = train_dataset.filler2id["あー"]
        logits[:, a_id] -= 0.0  # まずは小さく
        huun_id = train_dataset.filler2id["ふうん"]
        logits[:, huun_id] -= 0.005

        probs = F.softmax(logits, dim=-1)  # [1, C]

        pred_id = probs.argmax(dim=-1).item()
        pred_prob = probs[0, pred_id].item()

        gt_id = labels.item()

        gt = test_dataset.id2filler[gt_id]
        pred = train_dataset.id2filler[pred_id]

        print(f"{idx:3d} | {gt:8s} | {pred:8s} | {pred_prob:.3f}")

        if pred_id == gt_id:
            correct += 1
        total += 1

# =====================
# 精度表示
# =====================
acc = correct / total
print("\nExact Match Accuracy:", acc)
