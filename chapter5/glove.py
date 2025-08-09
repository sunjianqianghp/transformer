import torch
from torch import nn, optim
from torch.utils.data import Dataset
from collections import defaultdict
from .constants import BOS_TOKEN, EOS_TOKEN, DEVICE
from tqdm.auto import tqdm
from .utils import load_reuters, get_loader, save_pretrained


class GloveDataset(Dataset):
    def __init__(self, corpus, vocab, context_size=2):
        # 记录词与上下文在给定语料中的共现次数
        self.cooccur_counts = defaultdict(float)
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        for sentence in tqdm(corpus, desc="Dataset Construction"):
            sentence = [self.bos]+sentence+[self.eos]
            for i in range(1, len(sentence)-1):
                w = sentence[i]
                left_contexts  = sentence[max(0, i-context_size):i]
                right_contexts = sentence[i+1:min(len(sentence), i+context_size)+1]
                # 共现次数随距离衰减: 1/d(w,c)
                for k, c in enumerate(left_contexts[::-1]): # 逆序排列
                    self.cooccur_counts[(w, c)] += 1/(k+1)
                for k, c in enumerate(right_contexts):
                    self.cooccur_counts[(w, c)] += 1/(k+1)
        
        self.data = [(w, c, count) for (w, c), count in self.cooccur_counts.items()]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]
    
    def collate_fn(self, examples):
        words = torch.tensor([ex[0] for ex in examples])
        contexts = torch.tensor([ex[1] for ex in examples])
        counts = torch.tensor([ex[2] for ex in examples])
        # words: shape=(batch_size, )
        # contexts: shape=(batch_size, )
        # counts: shape=(batch_size, )
        return (words, contexts, counts)
    

class GloveModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(GloveModel, self).__init__()
        # 词向量及偏置向量
        self.w_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.w_biases = nn.Embedding(vocab_size, 1)
        # 上下文向量及偏置向量
        self.c_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.c_biases = nn.Embedding(vocab_size, 1)

    def forward_w(self, words):
        '''
        words: shape=(batch_size, )
        '''
        w_embeds = self.w_embeddings(words) # shape=(batch_size, embedding_dim)
        w_biases = self.w_biases(words)     # shape=(batch_size, 1)
        return w_embeds, w_biases

    def forward_c(self, contexts):
        '''
        contexts: shape=(batch_size, )
        '''
        c_embeds = self.c_embeddings(contexts) # shape=(batch_size, embedding_dim)
        c_biases = self.c_biases(contexts)     # shape=(batch_size, 1)
        return c_embeds, c_biases
    
if __name__ == '__main__':
    m_max = 100
    alpha = 0.75
    context_size = 3
    batch_size = 1024
    embedding_dim = 10
    num_epochs = 10


    # 构建Glove训练数据集
    corpus, vocab = load_reuters()
    dataset = GloveDataset(corpus, vocab, context_size=context_size)
    data_loader = get_loader(dataset, batch_size=batch_size)

    model = GloveModel(len(vocab), embedding_dim=embedding_dim)
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(data_loader, desc=f"Training Epoch {epoch}"):
            # words: shape=(batch_size, )
            # contexts: shape=(batch_size, )
            # counts: shape=(batch_size, )
            words, contexts, counts = [x.to(DEVICE) for x in batch]
            # 提取批次的内词、上下文的向量表示及偏置向量
            # word_embeds: shape=(batch_size, embedding_dim)
            # word_biases: shape=(batch_size, 1)
            word_embeds, word_biases = model.forward_w(words)
            # context_embeds: shape=(batch_size, embedding_dim)
            # context_biases: shape=(batch_size, 1)
            context_embeds, context_biases = model.forward_c(contexts)
            # 回归目标值
            log_counts = torch.log(counts) # shape=(batch_size,)
            # 样本权重
            weight_factor = torch.clamp(torch.pow(counts/m_max, alpha), max=1.0) # shape=(batch_size,)
            optimizer.zero_grad()

            # 计算批次内每个样本的L2损失
            # loss: shape=(batch_size,)
            loss = (torch.sum(word_embeds*context_embeds, dim=1)+word_biases+context_biases-log_counts)**2
            # 样本加权损失
            wavg_loss = (weight_factor*loss).mean()
            wavg_loss.backward()
            optimizer.step()
            total_loss += wavg_loss.item()
        print(f"Loss: {total_loss:.2f}")

    # 合并词向量矩阵与上下文向量矩阵，作为最终的预训练词向量
    combined_embeds = model.w_embeddings.weiht + model.c_embeddings.weight 
    # 将词向量保存至glove.vec文件
    save_pretrained(vocab, combined_embeds.data, "glove.vec")





