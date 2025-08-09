import torch


BOS_TOKEN = "<bos>" # 句首标记
EOS_TOKEN = "<eos>" # 句尾标记
PAD_TOKEN = "<pad>" # 补齐标记
BOW_TOKEN = "<bow>" # begin of word
EOW_TOKEN = "<eow>" # end of word

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
