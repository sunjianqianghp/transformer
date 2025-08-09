from .constants import PAD_TOKEN, BOS_TOKEN, EOS_TOKEN
from ..utils.util import Vocab
from torch.utils.data import DataLoader
import torch

def save_pretrained(vocab, embeds, path):
    with open(path, 'w', encoding='utf-8') as f:
        # 记录词向量大小 
        f.write(f"{embeds.shape[0]} {embeds.shape[1]}\n")
        for idx, token in enumerate(vocab.idx_to_token):
            vec = " ".join([f"{x}" for x in embeds[idx] ])
            # 每行对应一个单词，以及由空格分割的词向量
            f.write(f"{token} {vec}\n")  

def load_pretrained(load_path):
    with open(load_path, "r") as fin:
        # 第一行为词向量大小, n: vocab_size, d:embedding_dim
        n, d = map(int, fin.readline().split())
        tokens = []
        embeds = []
        for line in fin:
            line = line.rstrip().split(' ')
            token, embed = line[0], list(map(float, line[1:]))
            tokens.append(token)
            embeds.append(embed)
        vocab = Vocab(tokens)
        embeds = torch.tensor(embeds, dtype=torch.float)
    return vocab, torch


def load_reuters():
    # 从NLTK中导入Reuters数据处理模块
    from nltk.corpus import reuters
    # 获取Reuters数据中的所有句子（已完成标记解析）
    text = reuters.sents()
    #（可选）将语料中的词转换为小写
    text = [[word.lower() for word in sentence] for sentence in text]
    # 构建词表，并传入预留标记
    vocab = Vocab.build(text, reserved_token=[PAD_TOKEN, BOS_TOKEN, EOS_TOKEN])
    # 利用词表将文本数据转换为id表示
    corpus = [vocab.convert_tokens_to_ids(sentence) for sentence in text]
    return corpus, vocab 

def get_loader(dataset, batch_size, shuffle=True):
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=shuffle
    )
    return data_loader

def knn(W, x, k):
    similarities = torch.matmul(x, W.transpose(1, 0)) / (torch.norm(W, dim=1) * torch.norm(x) + 1e-9)
    knn = similarities.topk(k=k)
    return knn.values.tolist(), knn.indices.tolist()

def find_similar_words(embeds, vocab, query, k=5):
    knn_values, knn_indices = knn(embeds, embeds[vocab[query]], k + 1)
    knn_words = vocab.convert_ids_to_tokens(knn_indices)
    print(f">>> Query word: {query}")
    for i in range(k):
        print(f"cosine similarity={knn_values[i + 1]:.4f}: {knn_words[i + 1]}")

def find_analogy(embeds, vocab, word_a, word_b, word_c):
    vecs = embeds[vocab.convert_tokens_to_ids([word_a, word_b, word_c])]
    x = vecs[2] + vecs[1] - vecs[0]
    knn_values, knn_indices = knn(embeds, x, k=1)
    analogies = vocab.convert_ids_to_tokens(knn_indices)
    print(f">>> Query: {word_a}, {word_b}, {word_c}")
    print(f"{analogies}")

word_analogy_queries = [["brother", "sister", "man"],
                        ["paris", "france", "berlin"]]
vocab, embeds = load_pretrained("glove.vec")
for w_a, w_b, w_c in word_analogy_queries:
    find_analogy(embeds, vocab, w_a, w_b, w_c)

from torch import nn
from torch import functional as F
from ..utils.util import Vocab

class MLP(nn.Module):
    def __init__(self, vocab:Vocab, pt_vocab:Vocab, pt_embeddings, hidden_dim, num_class):
        super(MLP, self).__init__()
        # 与预训练词向量维度保持一致
        embedding_dim = pt_embeddings.shape[1]
        # 词向量层
        vocab_size = len(vocab)
        self.embeddings = nn.EmbeddingBag(vocab_size, embedding_dim)
        self.embeddings.weight.data.uniform_(-0.1, 0.1)
        # 使用预训练词向量对词向量层进行初始化
        for idx, token in enumerate(vocab.idx_to_token):
            pt_idx = pt_vocab[token]
            # 只初始化预训练词典中存在的词
            # 对于未出现在预训练词典中的词，保留其随机初始化向量
            if pt_idx != pt_vocab.unk:
                self.embeddings.weight[idx].data.copy_(pt_embeddings[pt_idx])
        
        # 线性变换：词向量层->隐含层
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        # 线性变换：隐含层->输出层
        self.fc2 = nn.Linear(hidden_dim, num_class)
        # 使用ReLU激活函数
        self.activate = F.relu




