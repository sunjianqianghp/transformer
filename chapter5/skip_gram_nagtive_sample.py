import torch
from torch import nn
from torch.utils.data import Dataset
from torch import optim
from torch.nn import functional as F
from tqdm.auto import tqdm
from .constants import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, DEVICE
from .utils import load_reuters, get_loader, save_pretrained


# 数据
class SGNSDataset(Dataset):
    def __init__(self, corpus, vocab, context_size=2, n_negatives=5, ns_dist=None):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        self.pad = vocab[PAD_TOKEN]
        for sentence in tqdm(corpus, desc="Dataset Construction"):
            sentence = [self.bos]+sentence+[self.eos]
            for i in range(1, len(sentence)-1):
                # 模型输入：(w, context); 输出为0/1， 表示context是否为负样本
                w = sentence[i]
                left_context_index = max(0, i-context_size)
                right_context_index = min(len(sentence), i+context_size)
                context = sentence[left_context_index:i]+sentence[i+1:right_context_index]
                # 将长度不足context_size的context用self.pad进行补齐
                context += [self.pad]*(2*context_size - len(context)) 
                self.data.append((w, context))

        # 负样本数量
        self.n_negatives = n_negatives
        # 负采样分布：若参数ns_dist为None，则使用均匀分布(从词表中均匀采样)
        self.ns_dist = ns_dist if ns_dist else torch.ones(len(vocab))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]
    
    def collate_fn(self, examples):
        '''
        examples:一个batch中的样本
        '''
        words    = torch.tensor([ex[0] for ex in examples], dtype=torch.long) # shape: (batch_size, )
        contexts = torch.tensor([ex[1] for ex in examples], dtype=torch.long) # shape: (batch_size, 2*context_size)
        batch_size, context_size = contexts.shape
        neg_contexts = []
        # 对批次内的样本分别进行负采样
        for i in range(batch_size):
            # 保证负样本不包含当前样本中的context，将self.ns_dist中context索引处的概率置为0
            # 并将结果赋值给ns_dist，而self.ns_dist的值不变
            ns_dist = self.ns_dist.index_fill(dim=0, index=contexts[i], value=0.0)
            neg_contexts.append(
                torch.multinomial(ns_dist, self.n_negatives*context_size, replacement=True)
            )
        neg_contexts = torch.stack(neg_contexts, dim=0) # shape: (batch_size, self.n_negatives*2*context_size)
        return words, contexts, neg_contexts

# 模型
class SGNSModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SGNSModel, self).__init__()
        # 词向量
        self.w_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 上下文向量
        self.c_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward_w(self, words):
        '''
        words: shape=(batch_size, )
        '''
        w_embeds = self.w_embeddings(words) # shape: (batch_size, embedding_dim)
        return w_embeds
    
    def forward_c(self, contexts):
        '''
        contexts: shape=(batch_size, 2*context_size)
        '''
        c_embeds = self.c_embeddings(contexts) # shape=(batch_size, 2*context_size, embedding_dim)
        return c_embeds
    
# 训练
def get_unigram_distribution(corpus, vocab_size):
    # 从给定语料中计算Unigram概率分布
    token_counts = torch.tensor([0]*vocab_size)
    total_count = 0
    for sentence in corpus:
        total_count += len(sentence)
        for token in sentence:
            token_counts[token] += 1
    unigram_dist = torch.div(token_counts.float(), total_count)
    return unigram_dist

if __name__ == '__main__':
    embedding_dim = 10
    context_size = 3
    batch_size = 1024
    n_negatives = 5 # 负样本数量
    num_epochs = 10

    # 读取文本数量
    corpus, vocab = load_reuters()
    # 计算Unigram概率分布
    unigram_dist = get_unigram_distribution(corpus, len(vocab))
    # 根据Unigram概率分布计算负采样分布：p(w)**0.75
    negative_sampling_dist = unigram_dist ** 0.75
    negative_sampling_dist /= negative_sampling_dist.sum()

    # 构建SGNS训练数据集
    dataset = SGNSDataset(
        corpus,
        vocab,
        context_size=context_size,
        n_negatives=n_negatives,
        ns_dist=negative_sampling_dist
    )
    data_loader = get_loader(dataset, batch_size)
    model = SGNSModel(len(vocab), embedding_dim)
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(data_loader, desc="Trainging Epoch {epoch}"):
            # words: shape=(batch_size,)
            # contexts: shape=(batch_size, 2*context_size)
            # neg_contexts: shape=(batch_size, n_negatives*2*context_size)
            words, contexts, neg_contexts = [x.to(DEVICE) for x in batch]
            optimizer.zero_grad()
            batch_size = words.shape[0]
            # 分别提取批次内词、上下文和负样本的向量表示
            word_embeds = model.forward_w(words).unsqueeze(dim=2) # shape=(batch_size, embedding_dim, 1)
            context_embeds = model.forward_c(contexts)            # shape=(batch_size, 2*context_size, embedding_dim)
            neg_context_embeds = model.forward_c(neg_contexts)    # shape=(batch_size, n_negatives*2*context_size, embedding_dim)
            # 正样本的分类(对数)似然
            context_loss = F.logsigmoid(
                torch.bmm(context_embeds, word_embeds).squeeze(dim=2) # shape=(batch_size, 2*context_size)
            )
            context_loss = context_loss.mean(dim=1) # shape=(batch_size, )

            # 负样本的分类(对数)似然
            neg_context_loss = F.logsigmoid(
                torch.bmm(neg_context_embeds, word_embeds).squeeze(dim=2).neg() # shape=(batch_size, n_negatives*2*context_size)
            )
            neg_context_loss = neg_context_loss.view(batch_size, -1, n_negatives) # shape=(batch_size, 2*context_size, n_negatives)
            neg_context_loss = neg_context_loss.sum(dim=2) # shape=(batch_size, 2*context_size)
            neg_context_loss = neg_context_loss.mean(dim=1) # shape=(batch_size, )

            # 总体损失
            loss = -(context_loss+neg_context_loss).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Loss: {total_loss:.2f}")

    # 合并词向量矩阵与上下文向量矩阵，作为最终的预训练词向量
    combined_embeds = model.w_embeddings.weight + model.c_embeddings.weight 
    # 将词向量保存至sgns.vec文件
    save_pretrained(vocab, combined_embeds.data, "sgns.vec")



