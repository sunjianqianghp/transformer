from typing import List, Optional


from collections import defaultdict
# defaultdict 是dict（字典）的一个子类，可以为字典中的每个键提供一个默认值。
# 如果访问字典中不存在的键，defaultdict 会自动为该键生成一个默认值，而不是抛出 KeyError。
# 用法： defaultdict(default_factory)， default_factory是一个函数，用来为不存在的键生成默认值。
#       例如，int 可以用来生成默认值 0，list 可以用来生成默认值空列表 []，等等
#       示例：
#            from collections import defaultdict
#            d = defaultdict(int)
#            d['apple'] += 1
#            d['banana'] += 2
#            print(d)  # 输出: defaultdict(<class 'int'>, {'apple': 1, 'banana': 2})

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class Vocab:
    def __init__(self, tokens:Optional[List] =None):
        self.idx_to_token = list()
        self.token_to_idx = dict()

        if tokens is not None:
            if "<unk>" not in tokens:
                tokens = tokens + ["<unk>"]
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token)-1
            self.unk = self.token_to_idx["<unk>"]

    @classmethod
    def build(cls, text:List, min_freq:int=1, reserved_token:Optional[List]=None):
        token_freqs = defaultdict(int)
        for sentence in text:
            for token in sentence:
                token_freqs[token] += 1
        uniq_tokens = ["<unk>"] + (reserved_token if reserved_token else [])
        uniq_tokens += [token for token, freq in token_freqs.items() if freq >= min_freq and token != "<unk>"]
        return cls(uniq_tokens)
    
    def __len__(self):
        # 返回词表的大小，即词表中有多少个互不相同的标记
        return len(self.idx_to_token)
    
    def __getitem__(self, token): 
        # 查找输入标记对应的索引值， 如果该标记不存在， 则返回标记<unk>的索引值（0）
        return self.token_to_idx.get(token, self.unk)
    
    def convert_tokens_to_ids(self, tokens):
        # 查找一系列输入标记对应的索引值
        return [self[token] for token in tokens]
    
    def convert_ids_to_tokens(self, indices):
        # 查找一系列索引值对应的标记
        return [self.idx_to_token[index] for index in indices]

def load_sentence_polarity():
    from nltk.corpus import sentence_polarity

    # 使用全部句子集合（已经过标记解析）创建词表
    vocab = Vocab.build(sentence_polarity.sents())

    # 褒贬各4000作为训练数据
    train_data = [(vocab.convert_tokens_to_ids(sentence), 0)
                  for sentence in sentence_polarity.sents(categories='pos')[:4000]] \
        + [(vocab.convert_tokens_to_ids(sentence), 1)
            for sentence in sentence_polarity.sents(categories='neg')[:4000]]

    # 其余的数据作为测试数据
    test_data = [(vocab.convert_tokens_to_ids(sentence), 0)
                 for sentence in sentence_polarity.sents(categories='pos')[4000:]] \
        + [(vocab.convert_tokens_to_ids(sentence), 1)
            for sentence in sentence_polarity.sents(categories='neg')[4000:]]

    return train_data, test_data, vocab

class BowDataset(Dataset):
    def __init__(self, data):
        # data为原始的数据，如load_sentence_polarity函数获得的训练数据
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]

def collate_fn_mlp(examples):
    '''
    params:
        examples: dataset
    '''
    # 从独立样本集合中构建各批次的输入输出
    # 其中，BowDataset类定义了一个样本的数据结构，即输入标签和输出标签的元组
    # 因此，将输入inputs定义为一个张量的列表，其中每个张量为原始句子中标记序列
    # 对应的索引值序列（ex[0]）
    inputs = [torch.tensor(ex[0]) for ex in examples]

    offsets = [0] + [i.shape[0] for i in inputs]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    inputs = torch.cat(inputs)
    return inputs, offsets, targets

class MLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class):
        super(MLP, self).__init__()
        # nn.EmbeddingBag, “词袋”嵌入聚合层‌。在检索词向量的同时，‌直接对一组索引的嵌入结果进行聚合‌（如均值(默认)、求和、最大值），
        # 避免中间嵌入实例化， 
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.activate = F.relu
        self.linear2 = nn.Linear(hidden_dim, num_class)
    
    def forward(self, inputs, offsets):
        embedding = self.embedding(inputs, offsets)
        hidden = self.activate(self.linear1(embedding))
        outputs = self.linear2(hidden)
        log_probs = F.log_softmax(outputs, dim=1)
        return log_probs


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, filter_size, num_filter, num_class):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1d = nn.Conv1d(
            in_channels=embedding_dim, 
            out_channels=num_filter, 
            kernel_size=filter_size, 
            stride=1, 
            padding=1
        ) # padding=1表示在卷积操作之前，将序列的前后各补充1个输入

        self.activate = F.relu 
        self.linear = nn.Linear(num_filter, num_class)

    def forward(self, inputs): 
        '''
        params:
            inputs: shape=(batch_size, seq_len)
        '''
        embedding:torch.Tensor = self.embedding(inputs) # shape=(batch_size, seq_len, embedding_dim)
        convolution = self.activate(
            self.conv1d(
                embedding.permute(0, 2, 1) # shape=(batch_size, embedding_dim, seq_len)
            ) # shape=(batch_size, num_filter, (seq_len+2*padding - (filter_size-stride) )//stride )
        )
        pooling = F.max_pool1d(convolution, kernel_size=convolution.shape[2]) # shape=(batch_size, num_filter, 1)
        outputs = self.linear(
            pooling.squeeze(dim=2) # shape=(batch_size, num_filter)
        )  # shape=(batch_size, 2)
        log_probs = F.log_softmax(outputs, dim=1)
        return log_probs


def collate_fn_cnn(examples):
    # inputs：列表，长度为batch_size, 每个元素为一个1D tensor，shape=（seqLength[i], ）, 其中0<=i<batch_size
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)

    # 对批次内的样本用0补齐， 使其具备相同长度
    inputs = pad_sequence(inputs, batch_first=True) # shape = (batch_size, max(seqLength)), 若batch_size=False, 则shape=(max(seqLength)， batch_size)
    return inputs, targets


from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class):
        super(LSTM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, num_class)

    def forward(self, inputs, lengths):
        # print('lengths.sum', lengths.sum())
        embeddings = self.embeddings(inputs) # shape=(batch_size, seq_len, embedding_dim) ， inputs已被填充补齐

        # print('embeddings', embeddings.shape)

        # 使用pack_padded_sequence函数将变长序列打包, 将batch_size个样本揉在一起
        x_pack:PackedSequence = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False) # x_pack.data.shape = (lengths.sum(), embedding_dim)
        # print('x_pack', len(x_pack), len(x_pack[0]), len(x_pack[1]), len(x_pack[2]), len(x_pack[3]))
        # print('x_pack', x_pack.data.shape)

        hidden, (hn, cn) = self.lstm(x_pack) # hidden.data.shape=(lengths.sum(), hidden_dim),  hn.shape=（1, batch_size, hidden_dim）, cn.shape=（1, batch_size, hidden_dim）,

        # print('hidden', hidden.data.shape)
        # print('hn', hn.shape)
        # print('cn', cn.shape)

        outputs = self.output(hn[-1]) # shape=(batch_size, num_class)
        # print('outputs', outputs.shape)
        log_probs = F.log_softmax(outputs, dim=-1)
        return log_probs
        

    
def collate_fn_lstm(examples):
    # 获得每个序列的长度
    lengths = torch.tensor([len(ex[0]) for ex in examples])

    # inputs：列表，长度为batch_size, 每个元素为一个1D tensor，shape=（seqLength[i], ）, 其中0<=i<batch_size
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)

    # 对批次内的样本用0补齐， 使其具备相同长度
    inputs = pad_sequence(inputs, batch_first=True) # shape = (batch_size, max(seqLength)), 若batch_first=False, 则shape=(max(seqLength)， batch_size)
    return inputs, lengths, targets


import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


def length_to_mask(lengths):
    max_len = torch.max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(lengths.shape[0], max_len) < lengths.unsqueeze(1)
    return mask


class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class,
                 dim_feedforward=512, num_head=2, num_layers=2, dropout=0.1, 
                 max_len=512, activation: str = "relu"):
        super(Transformer, self).__init__()
        # 词嵌入层
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim) # 词向量层 
        self.position_embedding = PositionalEncoding(embedding_dim, dropout, max_len)

        # 编码层：使用Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_head, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            activation=activation)
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出层
        self.output = nn.Linear(hidden_dim, num_class)

    def forward(self, inputs, lengths):
        '''
        params:
            inputs: shape=(batch_size,  max(seqLength))
        '''
        inputs = torch.transpose(inputs, 0, 1) # shape = (max(seqLength), batch_size)
        # 与LSTM处理情况相同，输入数据的第1维是批次，需要转换为TransformerEncoder所需要的第一维是长度，第二维是批次的形状

        hidden_states = self.embeddings(inputs)                # shape = (max(seqLength), batch_size, embedding_dim)
        
        hidden_states = self.position_embedding(hidden_states) # shape = (max(seqLength), batch_size, embedding_dim)
        attention_mask = length_to_mask(lengths) == False
        
        hidden_states = self.transformer(hidden_states, src_key_padding_mask=attention_mask) # shape = (max(seqLength), batch_size, hidden_dim)

        hidden_states = hidden_states[0, :, :] # shape = (batch_size, hidden_dim)

        output = self.output(hidden_states) # shape = (batch_size, num_class)
        
        log_probs = F.log_softmax(output, dim=1)
        
        return log_probs


# ============================================== 词性标注 Start ===============================================
def load_treebank():
    from nltk.corpus import treebank 
    # sents存储全部经过标记化的句子
    # postags存储每个标记对应的词性标注结果
    # sents, postags = zip()

    sents, postags = zip(*(zip(*sent) for sent in treebank.tagged_sents() )) 

    # "<pad>"为预留的用于补齐序列长度的标记
    vocab:Vocab = Vocab.build(sents, reserved_tokens=["<pad>"])

    # 字符串表示的词性标注标签，也需要使用词表映射为索引值
    tag_vocab = Vocab.build(postags)

    # 前3000句作为训练数据
    train_data = [(vocab.convert_tokens_to_ids(sentence), tag_vocab.convert_tokens_to_ids(tags)) for sentence, tags in zip(sents[:3000], postags[:3000])]
    # 其余的作为测试数据
    test_data  = [(vocab.convert_tokens_to_ids(sentence), tag_vocab.convert_tokens_to_ids(tags)) for sentence, tags in zip(sents[3000:], postags[3000:])]

    return train_data, test_data, vocab, tag_vocab

global vocab

def collate_fn(examples):
    lengths = torch.tensor([len(ex[0]) for ex in examples])

    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = [torch.tensor(ex[1]) for ex in examples]
    
    inputs = pad_sequence(inputs, batch_first=True, padding_value=vocab["<pad>"])
    targets = pad_sequence(targets, batch_first=True, padding_value=vocab["<pad>"])
    
    # 返回结果最后一项为mask项， 用于记录哪些是序列实际的有效标记
    return inputs, lengths, targets, inputs != vocab["<pad>"]

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class):
        super(LSTM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, num_class)

    def forward(self, inputs, lengths):
        embeddings = self.embeddings(inputs)
        x_pack = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        hidden, (hn, cn) = self.lstm(x_pack)
        # pad_packed_sequence函数与pack_padded_sequence相反， 是对打包的序列进行解包， 即还原成结尾经过补齐的多个序列
        hidden, _ = pad_packed_sequence(hidden, batch_first=True)
        # 在文本分类中，仅使用最后一个状态的隐含层（hc）， 而在序列标注中，需要使用序列全部状态的隐含层（hidden）
        outputs = self.output(hidden)
        log_probs = F.log_softmax(outputs, dim=-1)
        return log_probs

# ==============================================  词性标注 End  ===============================================


def save_vocab(vocab, path):
    with open(path, 'w') as writer:
        writer.write("\n".join(vocab.idx_to_token)) # 一行一个token


def read_vocab(path):
    with open(path, 'r') as f:
        tokens = f.read().split('\n')
    return Vocab(tokens)
