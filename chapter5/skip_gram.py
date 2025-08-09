from torch.utils.data import Dataset
from torch import nn
import torch.nn.functional as F
from .constants import PAD_TOKEN, BOS_TOKEN, EOS_TOKEN
from tqdm.auto import tqdm


class SkipGramDataset(Dataset):
    def __init__(self, corpus, vocab, context_size=2):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        for sentence in tqdm(corpus, desc="Dataset Construction"):
            sentence = [self.bos] + sentence + [self.eos]

            for i in range(1, len(sentence) - 1):
                # 模型输入：当前词
                w = sentence[i]

                # 模型输出：一定上下文窗口大小内共现的词对
                left_context_index = max(0, i-context_size)
                right_context_index = min(len(sentence), i+context_size)
                context = sentence[left_context_index:i] + sentence[i+1:right_context_index+1]
                self.data.append([(w, c) for c in context])



class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        # 词嵌入层
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 输出层
        self.output = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, inputs):
        '''
        inputs: (batch_size, )
        '''
        embeds = self.embeddings(inputs)  # shape: (batch_size, embedding_dim)
        # 根据当前词的词向量，对上下文进行预测（分类）
        output = self.output(embeds)  # shape: (batch_size, vocab_size)
        log_probs = F.log_softmax(output, dim=1)  # shape: (batch_size, vocab_size)
        return log_probs
