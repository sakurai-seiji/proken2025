import torch
import torch.nn as nn
from transformers import HubertModel

class HubertCTC(nn.Module):
    def __init__(
        self,
        model_name,
        num_classes,
        proj_dim=256,
        lstm_hidden=256,
        lstm_layers=1,
        dropout=0.1,
    ):
        super().__init__()

        # ===== HuBERT =====
        self.hubert = HubertModel.from_pretrained(model_name)

        # HuBERT freeze
        for p in self.hubert.parameters():
            p.requires_grad = False

        hubert_hidden = self.hubert.config.hidden_size

        # ===== Frame-wise projection =====
        self.frame_proj = nn.Linear(hubert_hidden, proj_dim)

        # ===== BiLSTM =====
        self.blstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # ===== Final classifier =====
        self.classifier = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, wavs, wav_lens):
        """
        wavs: [B, T]
        wav_lens: [B]  (HuBERT後のフレーム長を想定)
        """

        # ----- HuBERT -----
        feats = self.hubert(wavs).last_hidden_state  # [B, T', H]

        # ----- Linear (frame-wise) -----
        feats = self.frame_proj(feats)  # [B, T', proj_dim]

        # ----- BiLSTM -----
        lstm_out, _ = self.blstm(feats)  # [B, T', 2*lstm_hidden]

        # ----- padding-aware average pooling -----
        T = lstm_out.size(1)
        mask = torch.arange(T, device=wavs.device)[None, :] < wav_lens[:, None]
        mask = mask.unsqueeze(-1)  # [B, T', 1]

        pooled = (lstm_out * mask).sum(dim=1) / mask.sum(dim=1)  # [B, 2*lstm_hidden]

        # ----- Final linear -----
        logits = self.classifier(pooled)  # [B, num_classes]

        return logits
