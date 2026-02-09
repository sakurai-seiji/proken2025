import json
import torch
import soundfile as sf
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor

class SpeechDataset(Dataset):
    def __init__(self, manifest, model_name):
        self.items = [json.loads(l) for l in open(manifest)]
        self.extractor = AutoFeatureExtractor.from_pretrained(model_name)

        # ===== 相槌 → クラスID 辞書を作る =====
        fillers = sorted(set(item["filler"] for item in self.items))
        self.filler2id = {f: i for i, f in enumerate(fillers)}
        self.id2filler = {i: f for f, i in self.filler2id.items()}
        self.num_classes = len(self.filler2id)

        self.class_counts = [0] * self.num_classes
        for item in self.items:
            label_id = self.filler2id[item["filler"]]
            self.class_counts[label_id] += 1

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        wav, sr = sf.read(item["audio_filepath"])
        if wav.ndim == 2:
            wav = wav.mean(axis=1)

        inputs = self.extractor(
            wav,
            sampling_rate=sr,
            return_tensors="pt"
        ).input_values[0]   # [T]

        label_id = torch.tensor(
            self.filler2id[item["filler"]],
            dtype=torch.long
        )

        return inputs, label_id
