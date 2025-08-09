from torch.utils.data import Dataset
from torch import nn
import torch.nn.functional as F
from .constants import PAD_TOKEN, BOS_TOKEN, EOS_TOKEN
from tqdm.auto import tqdm


class CbowDataset(Dataset):
    def __init__(self, corpus, vocab, context_size=2):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        for sentence in tqdm(corpus, desc="Dataset Construction"):
            sentence = [self.bos] + sentence + [self.eos]
            if len(sentence) < context_size * 2 + 1: # 如果句子长度不足以构建(上下文、 目标词)训练样本，则跳过
                continue
            for i in range(context_size, len(sentence) - context_size):
                # 模型输入：左右分别取context_size长度的上文和下文
                context = sentence[i - context_size:i] + sentence[i+1:i + context_size + 1]
                # 模型输出：当前词
                target = sentence[i]
                self.data.append((context, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

class CbowModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CbowModel, self).__init__()
        # 词嵌入层
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 输出层 
        self.output = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, inputs):
        '''
        inputs: (batch_size, context_size * 2)
        '''
        embeds = self.embeddings(inputs) # shape: (batch_size, context_size * 2, embedding_dim)
        # 计算隐含层： 对上下文词向量求平均
        hidden = embeds.mean(dim=1) # shape: (batch_size, embedding_dim)
        output = self.output(hidden) # shape: (batch_size, vocab_size)
        log_probs = F.log_softmax(output, dim=1) # shape: (batch_size, vocab_size)
        return log_probs


