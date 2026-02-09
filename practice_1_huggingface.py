from transformers import HubertModel
import torch.nn as nn

hubert = HubertModel.from_pretrained("yky-h/japanese-hubert-base")

for p in hubert.parameters():
    p.requires_grad = False


import string

vocab = list("ぁあぃいぅうぇえぉおかきくけこさしすせそ"
             "たちつてとなにぬねのはひふへほ"
             "まみむめもやゆよらりるれろわをん"
             "ァアィイゥウェエォオカキクケコサシスセソ"
             "タチツテトナニヌネノハヒフヘホ"
             "マミムメモヤユヨラリルレロワヲンー")
vocab += list(string.ascii_lowercase)
vocab += list(string.ascii_uppercase)
vocab += list("0123456789 ")

# CTC blank を 0 にする
vocab_size = len(vocab) + 1


ctc_head = nn.Linear(hubert.config.hidden_size, vocab_size)
