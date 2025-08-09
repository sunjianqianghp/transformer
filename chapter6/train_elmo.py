import os
from os.path import join
import json
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch import optim
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
import numpy as np
from tqdm.auto import tqdm
from .constants import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, BOW_TOKEN, EOW_TOKEN, DEVICE
from ..utils.util import Vocab, save_vocab, read_vocab
from ..chapter5.utils import get_loader


def load_corpus(path, max_tok_len=None, max_seq_len=None):
    '''
    从生文本语料中加载数据并构建词表
    max_tok_len: 词的长度（字符数）上限
    max_seq_len: 序列长度（词数）上限
    '''
    text = []
    # 字符集，首先加入预定义特殊标记，包括“句首”、“句尾”、“补齐标记”、“词首”和“词尾”
    charset = {BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, BOW_TOKEN, EOW_TOKEN}
    print(f"Loading corpus from {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            tokens = line.rstrip().split(" ")
            # 截断长序列
            if max_seq_len is not None and len(tokens) + 2 > max_seq_len:
                tokens = line[:max_seq_len-2]
            
            sent = [BOS_TOKEN] # 添加句首
            for token in tokens:
                # 截断字符数目过多的词
                if max_tok_len is not None and len(token) + 2 > max_tok_len:
                    token = token[:max_tok_len-2]
                sent.append(token)
                for ch in token:
                    charset.add(ch)
            sent.append(EOS_TOKEN) # 添加句尾

            text.append(sent) 

    # Build word vocabulary(词表)
    vocab_w = Vocab.build(
        text,
        min_freq=2,
        reserved_tokens=[PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]
    )
    # Build character vocabulary(词表)
    vocab_c = Vocab(tokens=list(charset))

    # 构建词级语料(corpus)
    corpus_w = [vocab_w.convert_tokens_to_ids(sent) for sent in text] # LIST[LIST[int]]

    # 构建字符级语料
    corpus_c = [] # LIST[LIST[LIST[int]]]
    bow = vocab_c[BOW_TOKEN] # id
    eow = vocab_c[EOW_TOKEN] # id
    for i, sent in enumerate(text):
        sent_c = [] # LIST[LIST[int]]
        for token in sent:
            if token == BOS_TOKEN or token == EOS_TOKEN:
                token_c = [bow, vocab_c[token], eow] # LIST[int]
            else:
                token_c = [bow] + vocab_c.convert_tokens_to_ids(token) + [eow]
            sent_c.append(token_c)
        assert len(sent_c) == len(corpus_w[i])
        corpus_c.append(sent_c)

    assert len(corpus_w) == len(corpus_c)
    return corpus_w, corpus_c, vocab_w, vocab_c

# Dataset
class BiLMDataset(Dataset):
    def __init__(self, corpus_w, corpus_c, vocab_w, vocab_c):
        super(BiLMDataset, self).__init__()
        self.pad_w = vocab_w[PAD_TOKEN] # id
        self.pad_c = vocab_c[PAD_TOKEN] # id

        self.data = [] # LIST[(     LIST[int],    LIST[LIST[int]]    )]
        for sent_w, sent_c in tqdm(zip(corpus_w, corpus_c)):
            self.data.append((sent_w, sent_c))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate_fn(self, examples):
        # 当前批次中各样本序列的长度
        seq_lens = torch.LongTensor([len(ex[0]) for ex in examples])



        # 词级别输入：batch_size*seq_len(variable)
        inputs_w = [torch.tensor(ex[0]) for ex in examples]
        
        # 对batch内的样本进行长度补齐
        # inputs_w: shape=(batch_size, max_seq_len)
        inputs_w = pad_sequence(inputs_w, batch_first=True, padding_value=self.pad_w)


        # 计算当前批次中的最大序列长度
        batch_size, max_seq_len = inputs_w.shape
        # 计算当前批次中单词的最大字符数目
        max_tok_len = max([max([len(tok) for tok in ex[1]]) for ex in examples])

        # 字符级别输入：batch_size*max_seq_len*max_tok_len
        inputs_c = torch.LongTensor(batch_size, max_seq_len, max_tok_len).fill_(self.pad_c)
        for i, (sent_w, sent_c) in enumerate(examples):
            for j, tok in enumerate(sent_c): # tok: LIST[int]
                inputs_c[i][j][:len(tok)] = torch.LongTensor(tok)

        # 前向、后向语言模型的目标输出序列
        targets_fw = torch.LongTensor(inputs_w.shape).fill_(self.pad_w) # batch_size*max_seq_len
        targets_bw = torch.LongTensor(inputs_w.shape).fill_(self.pad_w) # batch_size*max_seq_len
        for i, (sent_w, sent_c) in enumerate(examples):
            targets_fw[i][:len(sent_w)-1] = torch.LongTensor(sent_w[1:])
            targets_bw[i][1:len(sent_w)]  = torch.LongTensor(sent_w[:len(sent_w)-1])

        return inputs_w, inputs_c, seq_lens, targets_fw, targets_bw


# Model Components
class Highway(nn.Module):
    def __init__(self, input_dim, num_layers, activation=F.relu):
        super(Highway, self).__init__()
        self.input_dim = input_dim
        self.layers = torch.nn.ModuleList(
            [nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)]
        )
        self.activation = activation
        for layer in self.layers:
            # layer.bias: shape = (2*input_dim,)
            # set bias in the gates to be positive
            # such that the highway layer will be biased towards the input part
            layer.bias[input_dim:].data.fill_(1)

    def forward(self, inputs):
        '''
        inputs: shape=(batch_size, input_dim)
        '''
        curr_inputs = inputs
        for layer in self.layers:
            projected_inputs = layer(curr_inputs) # (batch_size, 2*input_dim)
            # 输出向量的前半部分作为当前隐含层的输出
            hidden = self.activation(projected_inputs[:, 0:self.input_dim]) # (batch_size, input_dim)
            # 后半部分用于计算门控向量
            gate   =   torch.sigmoid(projected_inputs[:, self.input_dim: ]) # (batch_size, input_dim)
            # 线性插值
            curr_inputs = gate * curr_inputs + (1 - gate) * hidden          # (batch_size, input_dim)
        return curr_inputs


class ConvTokenEmbedder(nn.Module):
    '''
    vocab_c: 字符级词表
    char_embedding_dim: 字符向量维度
    char_conv_filters: 卷积核大小， [(kernel_size, out_channels), ...]
    num_highways: Highway网络层数
    output_dim: 输出层词向量维度
    '''
    def __init__(self, vocab_c, char_embedding_dim, char_conv_filters, 
                 num_highways, output_dim, pad="<pad>"):
        super(ConvTokenEmbedder, self).__init__()
        self.vocab_c = vocab_c

        self.char_embeddings = nn.Embedding(
            len(vocab_c),
            char_embedding_dim,
            padding_idx=vocab_c[pad] # 该索引对应的嵌入将始终是全0，并且不参与训练（默认 None）
        )
        self.char_embeddings.weight.data.uniform_(-0.25, 0.25)

        # 为每个卷积核分别构建卷积神经网络，这里使用一维卷积操作
        self.convolutions = nn.ModuleList()
        for kernel_size, out_channels in char_conv_filters:
            conv = torch.nn.Conv1d(
                in_channels=char_embedding_dim,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=0,
                bias=True
            )
            self.convolutions.append(conv)

        # 由多个卷积网络得到的向量表示拼接后的维度
        self.num_filters = sum(f[1] for f in char_conv_filters) 
        self.num_highways = num_highways
        self.highways = Highway(self.num_filters, self.num_highways, activation=F.relu)

        # 由于ELMo向量表示是多层表示的插值结果，因此需要保证各层向量表示的维度一致
        self.projection = nn.Linear(self.num_filters, output_dim, bias=True)

    def forward(self, inputs):
        '''
        inputs: BiLMDataset中的inputs_c, shape=(batch_size, max_seq_len, max_tok_len), 
               其中，max_tok_len是字符数目（不足的用'<pad>'对应的id进行填充）
        '''
        batch_size, seq_len, token_len = inputs.shape
        inputs = inputs.view(batch_size*seq_len, -1) # (batch_size*seq_len, token_len)

        # char_embeds: (batch_size*seq_len, token_len, char_embedding_dim)
        char_embeds = self.char_embeddings(inputs) 
        # char_embeds: (batch_size*seq_len, char_embedding_dim, token_len), char_embedding_dim为通道个数
        char_embeds = char_embeds.transpose(1, 2)  

        conv_hiddens = []
        for i in range(len(self.convolutions)):
            # conv_hidden: (batch_size*seq_len, out_channels_i, token_len-kernel_size_i+1)
            conv_hidden = self.convolutions[i](char_embeds)
            # conv_hidden: (batch_size*seq_len, out_channels_i)
            conv_hidden, _ = torch.max(conv_hidden, dim=-1) # 池化操作，token维度取最大
            # conv_hidden: (batch_size*seq_len, out_channels_i)
            conv_hidden = F.relu(conv_hidden)
            conv_hiddens.append(conv_hidden)

        # token_embeds: (batch_size*seq_len, self.num_filters)， self.num_filters=out_channels_0+...+out_channels_{n-1}
        token_embeds = torch.cat(conv_hiddens, dim=-1)
        # token_embeds: (batch_size*seq_len, self.num_filters)
        token_embeds = self.highways(token_embeds)
        # token_embeds: (batch_size*seq_len, output_dim)
        token_embeds = self.projection(token_embeds)
        # token_embeds: (batch_size, seq_len, output_dim)
        token_embeds = token_embeds.view(batch_size, seq_len, -1)
        return token_embeds


class ELMoLstmEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(ELMoLstmEncoder, self).__init__()
        # 保证LSTM各“中间层”及“输出层”具有和“输入表示层”相同的维度
        self.projection_dim = input_dim
        self.num_layers = num_layers
        # 前向LSTM(多层)
        self.forward_layers       = nn.ModuleList()
        # 前向LSTM投射层: hidden_dim -> self.projection_dim
        self.backward_layers      = nn.ModuleList()
        # 后向LSTM(多层)
        self.forward_projections  = nn.ModuleList()
        # 后向LSTM投射层: hidden_dim -> self.projection_dim
        self.backward_projections = nn.ModuleList()

        lstm_input_dim = input_dim
        for _ in range(num_layers):
            # 单层前向LSTM以及投射层
            forward_layer = nn.LSTM(
                lstm_input_dim,
                hidden_dim,
                num_layers=1,
                batch_first=True
            )
            forward_projection = nn.Linear(hidden_dim, self.projection_dim, bias=True)

            # 单层后向LSTM以及投射层
            backward_layer = nn.LSTM(
                lstm_input_dim,
                hidden_dim,
                num_layers=1,
                batch_first=True
            )
            backward_projection = nn.Linear(hidden_dim, self.projection_dim, bias=True)

            lstm_input_dim = self.projection_dim

            self.forward_layers.append(forward_layer)
            self.forward_projections.append(forward_projection)
            self.backward_layers.append(backward_layer)
            self.backward_projections.append(backward_projection)

    def forward(self, inputs, lengths):
        '''
        params:
            inputs: (batch_size, seq_len, input_dim)
            lengths:(batch_size, )
        '''
        batch_size, seq_len, input_dim = inputs.shape


        # rev_idx: 根据“前向输入批次”以及“批次中序列长度信息”，构建“后向输入批次”
        # 倒置序列索引，如[19, 7, 8, 0, 0, 0] -> [8, 7, 19, 0, 0, 0]
        # rev_idx: (batch_size, seq_len)
        rev_idx = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
        for i in range(lengths.shape[0]):
            rev_idx[i,:lengths[i]] = torch.arange(lengths[i]-1, -1, -1) # lengths[i]-1, lengths[i]-2, ..., 1, 0
        # rev_idx: (batch_size, seq_len, 1) -> (batch_size, seq_len, input_dim), 
        # 最后一维的数值重复了input_dim次
        rev_idx = rev_idx.unsqueeze(2).expand_as(inputs)
        rev_idx = rev_idx.to(inputs.device)
        rev_inputs = inputs.gather(1, rev_idx) # 见markdown


        # 前向、后向LSTM输入
        forward_inputs, backward_inputs = inputs, rev_inputs
        # 用于保存每一层前向、后向隐含层状态, 
        # stacked_forward_states和stacked_backward_states的元素数为num_layers
        stacked_forward_states, stacked_backward_states = [], []

        for layer_index in range(self.num_layers):
            # Transfer `lengths` to CPU to be compatible with latest PyTorch versions.
            packed_forward_inputs = pack_padded_sequence(
                forward_inputs, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_backward_inputs = pack_padded_sequence(
                backward_inputs, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

            # forward
            forward_layer = self.forward_layers[layer_index]
            # packed_forward:输出序列的隐含层
            # hn: 最后一个时刻的隐含层
            # cn: 最后一个时刻的记忆细胞
            packed_forward, (hn, cn) = forward_layer(packed_forward_inputs)
            # forward:(batch_size, seq_len, hidden_dim)
            forward, lengths = pad_packed_sequence(packed_forward, batch_first=True)
            # forward:(batch_size, seq_len, self.projection_dim)
            forward = self.forward_projections[layer_index](forward)
            stacked_forward_states.append(forward)

            # backward
            backward_layer = self.backward_layers[layer_index]
            packed_backward, _ = backward_layer(packed_backward_inputs)
            # forward:(batch_size, seq_len, hidden_dim)
            backward = pad_packed_sequence(packed_backward, batch_first=True)[0]
            # forward:(batch_size, seq_len, self.projection_dim)
            backward = self.backward_projections[layer_index](backward)
            # convert back to original sequence order using rev_idx
            stacked_backward_states.append(backward.gather(1, rev_idx))

            forward_inputs, backward_inputs = forward, backward

        # stacked_forward_states : [tensor(batch_size, seq_len, projection_dim)] * num_layers
        # stacked_backward_states: [tensor(batch_size, seq_len, projection_dim)] * num_layers
        return stacked_forward_states, stacked_backward_states


configs = {
    'max_tok_len': 50, # 单词的最大长度
    # 经过预处理的训练语料文件，每行是一段独立的文本
    'train_file': './train.txt', # path to your training file, line-by-line and tokenized
    'model_path': './elmo_bilm', # 模型保存目录
    'char_embedding_dim': 50,    # 字符向量维度
    # 卷积核列表，每个卷积核大小由 [宽度，输出通道数] 表示
    'char_conv_filters': [[1, 32], [2, 32], [3, 64], [4, 128], [5, 256], [6, 512], [7, 1024]],
    # Highway网络数
    'num_highways': 2,
    # 投射向量维度, ConvTokenEmbedder输出的词向量的维度，ELMoLstmEncoder输入的词向量维度
    'projection_dim': 512,
    # LSTM隐含层维度
    'hidden_dim': 4096,
    # LSTM层数
    'num_layers': 2,
    'batch_size': 32,
    'dropout_prob': 0.1,
    'learning_rate': 0.0004,
    # 梯度最大范数，用于训练过程中的梯度裁剪
    'clip_grad': 5,
    'num_epoch': 10
}


class BiLM(nn.Module):
    """
    多层双向语言模型。
    """
    def __init__(self, configs, vocab_w, vocab_c):
        super(BiLM, self).__init__()
        self.dropout_prob = configs['dropout_prob']
        self.num_classes = len(vocab_w)

        self.token_embedder = ConvTokenEmbedder(
            vocab_c,
            configs['char_embedding_dim'],
            configs['char_conv_filters'],
            configs['num_highways'],
            configs['projection_dim']
        )

        self.encoder = ELMoLstmEncoder(
            configs['projection_dim'],
            configs['hidden_dim'],
            configs['num_layers']
        )

        self.classifier = nn.Linear(configs['projection_dim'], self.num_classes)

    def forward(self, inputs, lengths):
        '''
        params:
            inputs: BiLMDataset中的inputs_c, shape=(batch_size, max_seq_len, max_tok_len), 
                    其中，max_tok_len是字符数目（不足的用'<pad>'对应的id进行填充）
            lengths:(batch_size, ), 批次中每个文本序列的有效长度
        '''
        # token_embeds: (batch_size, seq_len, projection_dim)
        token_embeds = self.token_embedder(inputs)
        # token_embeds: (batch_size, seq_len, projection_dim)
        token_embeds = F.dropout(token_embeds, self.dropout_prob)
        # forward : [tensor(batch_size, seq_len, projection_dim)] * num_layers
        # backward: [tensor(batch_size, seq_len, projection_dim)] * num_layers
        forward, backward = self.encoder(token_embeds, lengths)

        return self.classifier(forward[-1]), self.classifier(backward[-1]) # (batch_size, seq_len, self.num_classes)
        

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.token_embedder.state_dict(), join(path, 'token_embedder.pth'))
        torch.save(self.encoder.state_dict(),        join(path, 'encoder.pth'))
        torch.save(self.classifier.state_dict(),     join(path, 'classifier.pth'))

    def load_pretrained(self, path):
        self.token_embedder.load_state_dict(torch.load(join(path, 'token_embedder.pth')))
        self.encoder.load_state_dict(torch.load(join(path, 'encoder.pth')))
        self.classifier.load_state_dict(torch.load(join(path, 'classifier.pth')))

# 训练
# 首先，构建训练数据和加载器
corpus_w, corpus_c, vocab_w, vocab_c = load_corpus(configs['train_file'])
train_data = BiLMDataset(corpus_w, corpus_c, vocab_w, vocab_c)
train_loader = get_loader(train_data, configs['batch_size'])

# 交叉熵损失函数
criterion = nn.CrossEntropyLoss(
    ignore_index=vocab_w[PAD_TOKEN], # 忽略所有PAD_TOKEN处的预测损失
    reduction="sum"
)

# 创建模型并加载至相应设备
print("Building BiLM model")
model = BiLM(configs, vocab_w, vocab_c)
model.to(DEVICE)
model.get_outpu

optimizer = optim.Adam(
    filter(lambda x: x.requires_grad, model.parameters()),
    lr=configs['learning_rate']
)


if __name__ == '__main__':
    # 训练过程
    model.train()
    for epoch in range(configs['num_epoch']):
        total_loss = 0
        total_tags = 0 # 有效预测位置的数量，即非PAD_TOKEN处的预测
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            batch = [x.to(DEVICE) for x in batch]
            inputs_w, inputs_c, seq_lens, targets_fw, targets_bw = batch

            optimizer.zero_grad()
            # outputs_fw, outputs_bw: shape=(batch_size, seq_len, len(vocab_w))
            outputs_fw, outputs_bw = model(inputs_c, seq_lens)
            # 前向语言模型损失
            loss_fw = criterion(
                outputs_fw.view(-1, outputs_fw.shape[-1]), # shape=(batch_size*seq_len, len(vocab_w) )
                targets_fw.view(-1)
            )
            # 后向语言模型损失
            loss_bw = criterion(
                outputs_bw.view(-1, outputs_bw.shape[-1]), # shape=(batch_size*seq_len, len(vocab_w) )
                targets_bw.view(-1)
            )
            loss = (loss_fw + loss_bw) / 2.0
            loss.backward() # 反向求梯度
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), configs['clip_grad'])
            optimizer.step() # 迭代参数

            total_loss += loss_fw.item()
            total_tags += seq_lens.sum().item()
        
        # 以前向语言模型的困惑度(PPL)作为模型当前性能指标
        train_ppl = np.exp(total_loss / total_tags)
        print(f"Train PPL: {train_ppl:.2f}")

    # save BiLM encoders
    model.save_pretrained(configs['model_path'])
    # save configs
    json.dump(configs, open(join(configs['model_path'], 'configs.json'), "w"))

    # save vocabularies
    save_vocab(vocab_w, join(configs['model_path'], 'word.dic'))
    save_vocab(vocab_c, join(configs['model_path'], 'char.dic'))

# 封装编码器部分
class ELMo(nn.Module):
    def __init__(self, model_dir):
        super(ELMo, self).__init__()
        # 加载配置文件，获得模型超参数
        self.configs = json.load(open(join(model_dir, 'configs.json')))

        # 读取词表，此处只须读取字符级词表
        self.vocab_c = read_vocab(join(model_dir, 'char.dic'))

        # 词表示编码器
        self.token_embedder = ConvTokenEmbedder(
            self.vocab_c,
            self.configs['char_embedding_dim'],
            self.configs['char_conv_filters'],
            self.configs['num_highways'],
            self.configs['projection_dim'],
        )

        # Elmo LSTM编码器
        self.encoder = ELMoLstmEncoder(
            self.configs['projection_dim'],
            self.configs['hidden_dim'],
            self.configs['num_layers']
        )

        self.output_dim = self.configs['projection_dim']

        # 从预训练模型模型目录中加载编码器
        self.load_pretrained(model_dir)

    def get_output_dim(self):
        return self.output_dim

    def load_pretrained(self, path):
        # 加载词表示编码器, .load_state_dict为继承自nn.Module的方法
        self.token_embedder.load_state_dict(
            torch.load(join(path, "token_embedder.pth"))
        )
        # 加载编码器
        self.encoder.load_state_dict(
            torch.load(join(path, "encoder.pth"))
        )







