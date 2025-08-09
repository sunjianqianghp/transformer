import torch


BOS_TOKEN = "<bos>" # 句首标记
EOS_TOKEN = "<eos>" # 句尾标记
PAD_TOKEN = "<pad>" # 补齐标记

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')