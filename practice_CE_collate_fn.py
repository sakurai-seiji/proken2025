import torch

def collate_fn(batch):
    wavs, labels = zip(*batch)

    wav_lens = torch.tensor([len(w) for w in wavs], dtype=torch.long)
    wavs = torch.nn.utils.rnn.pad_sequence(
        wavs, batch_first=True
    )

    labels = torch.stack(labels)  # [B]

    return wavs, wav_lens, labels
