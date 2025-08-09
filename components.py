import math
from typing import Union, List, Optional
from torch import nn
import torch
from d2l import torch as d2l


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        '''
        params:
            num_hiddens: 每个位置编码向量的长度；
        '''
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
  

def sequence_mask(X:torch.Tensor, valid_len:torch.Tensor, value=0):
    """在序列中屏蔽不相关的项
    params:
        X:         torch.Tensor, shape=(batch_size*num_heads*query_size, key_size) 
        valid_len: torch.Tensor, shape=(batch_size*num_heads*query_size, )
    """
    key_size = X.size(1) 
    # [None,:]的作用等价于“.unsqueeze(0)”, shape=(batch_size*num_heads*query_size, key_size)
    #      shape=(1, key_size),                                                   shape=(batch_size*num_heads*query_size, 1)
    mask = torch.arange(key_size, dtype=torch.float32, device=X.device)[None,:] < valid_len[:, None]
    X[~mask] = value
    return X

def masked_softmax(X, valid_lens=None):
    """通过在最后一个轴上的掩码元素来执行softmax操作
    params:
        X:          shape = (batch_size*num_heads, query_size, key_size);
        valid_lens: shape = (batch_size*num_heads, ) or (batch_size*num_heads, query_size)  or (batch_size*num_heads, num_steps);
    """
    # print(f'masked_softmax 0, X.shape:{X.shape}, valid_lens.shape:{valid_lens.shape}')
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape # = (batch_size, query_size, key_size)
        if valid_lens.dim() == 1:
            # 如valid_lens = [num_heads个3, num_heads个2, num_heads个4], shape[1](query_size), 则repeat后变为
            # [num_heads*query_size个3, num_heads*query_size个2, num_heads*query_size个4]
            valid_lens = torch.repeat_interleave(valid_lens, shape[1]) # shape = (batch_size*num_heads*query_size, )
        else:
            # 如valid_lens = [num_heads个[1,2,3, ..., num_steps],
            #                 num_heads个[1,2,3, ..., num_steps],
            #                 num_heads个[1,2,3, ..., num_steps]]  shape = (batch_size*num_heads*num_steps, )
        # reshape后变为 [num_heads*query_size个3, num_heads*query_size个2, num_heads*query_size个4]
        #           or  [batch_size*num_heads个1, 2, 3, ..., num_steps]
            valid_lens = valid_lens.reshape(-1)

        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        #                (batch_size*num_heads*query_size, key_size)
        # print(f'masked_softmax 1, X.shape:{X.shape}, valid_lens.shape:{valid_lens.shape}')
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

class DotProductAttention(nn.Module):
    def __init__(self, dropout=0.1, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        '''
            queries:    shape = (batch_size*num_heads,  query_size, num_hiddens/num_heads);
            keys:       shape = (batch_size*num_heads,    key_size, num_hiddens/num_heads);
            values:     shape = (batch_size*num_heads,  value_size, num_hiddens/num_heads);
            valid_lens: shape = (batch_size*num_heads, ) or (batch_size*num_heads, query_size) or (batch_size*num_heads, num_steps);
            其中， num_hiddens是num_heads的倍数 
        '''
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2))/math.sqrt(d) # shape=(batch_size*num_heads, query_size, key_size)
        # print('scores.shape', scores.shape)
        self.attention_weights = masked_softmax(scores, valid_lens)    # shape=(batch_size*num_heads, query_size, key_size)
        return torch.bmm(self.dropout(self.attention_weights), values) # shape=(batch_size, query_size, num_hiddens)


def transpose_qkv(X:torch.Tensor, num_heads:int):
    """为了多注意力头的并行计算而变换形状
    params:
        X: shape = (batch_size, query_size or key_size, num_hiddens)
    """
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1) # shape=(batch_size, query_size or key_size, num_heads, num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)                            # shape=(batch_size, num_heads, query_size or key_size, num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])         # shape=(batch_size*num_heads,  query_size or key_size, num_hiddens/num_heads)

def transpose_output(X:torch.Tensor, num_heads:int):
    """逆转transpose_qkv函数的操作
    params:
        X: shape=(batch_size*num_heads, query_size, num_hiddens/num_heads)
    """
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2]) # shape=(batch_size, num_heads, query_size, num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3) # shape=(batch_size, query_size, num_heads, num_hiddens/num_heads)
    return X.reshape(X.shape[0], X.shape[1], -1) # shape=(batch_size, query_size, num_hiddens)


class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, 
                 dim_key, 
                 dim_query, 
                 dim_value, 
                 num_hiddens, 
                 num_heads, 
                 dropout=0.1, 
                 bias=False, 
                 **kwargs):
        '''
        params:
            key_size  :  教材中的d_k;
            query_size:  教材中的d_q;
            value_size:  教材中的d_v;
            num_hiddens: 教材中的p_o;
            num_heads:   教材中的h;
        关系： p_q = p_k = p_v = p_o/h
        '''
        assert num_hiddens % num_heads == 0
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.dp_attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(dim_query,  num_hiddens, bias=bias) # self.W_q(queries) shape=(batch_size, query_size, num_hiddens)
        self.W_k = nn.Linear(  dim_key,  num_hiddens, bias=bias)
        self.W_v = nn.Linear(dim_value,  num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        '''
        params:
               queries: shape=(batch_size, query_size, dim_query)
                  keys: shape=(batch_size,   key_size,   dim_key)
                values: shape=(batch_size, value_size, dim_value)
            valid_lens: shape=(batch_size, ) or (batch_size, query_size)

                  e.g.: [3, 2, 4] or [[1,2,3,...,num_steps], [1,2,3,...,num_steps], [1,2,3,...,num_steps]]
        '''
        queries = transpose_qkv(self.W_q(queries), self.num_heads) # shape = (batch_size*num_heads,  query_size, num_hiddens/num_heads)
        keys    = transpose_qkv(   self.W_k(keys), self.num_heads) # shape = (batch_size*num_heads,  num_kv,     num_hiddens/num_heads)
        values  = transpose_qkv( self.W_v(values), self.num_heads) # shape = (batch_size*num_heads,  num_kv,     num_hiddens/num_heads)

        if valid_lens is not None:
            # 将第一项（标量或矢量）复制num_heads次，然后如此重复第二项，依次类推
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
            '''[num_heads个3, num_heads个2, num_heads个4]   (batch_size*num_heads)
            or [num_heads个[1,2,3, ..., num_steps],         (batch_size*num_heads, num_steps)
                num_heads个[1,2,3, ..., num_steps],
                num_heads个[1,2,3, ..., num_steps]]
            '''
        
        output = self.dp_attention(queries, keys, values, valid_lens) # shape=（batch_size*num_heads, query_size, num_hiddens/num_heads)
        output_concat = transpose_output(output, self.num_heads)      # shape= (batch_size, query_size, num_hiddens)
        return self.W_o(output_concat)                                # shape= (batch_size, query_size, num_hiddens)


class AddNorm(nn.Module):
    """残差连接后进行规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        """
        params: 
            normalized_shape: 特征数
            dropout: float
        """
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout) 
        # LayerNorm 基于"特征"维度进行规范化
        # BatchNorm 基于"样本"维度进行规范化
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        # print( X.shape, Y.shape )
        return self.ln(self.dropout(Y)+X)


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        '''
        ffn是在最后一维上进行, 所以valid_lens的mask在encoder的各个子层都要应用
        '''
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    # pred:      shape=(batch_size, num_steps, vocab_size)
    # label:     shape=(batch_size, num_steps)
    # valid_len: shape=(batch_size)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
        pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


def train_seq2seq(net:nn.Module, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型"""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    
    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2) # 训练损失总和，词元数量
        for i, batch in enumerate(data_iter):
            # print(f'i:{i}')
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1) 

            Y_hat, _ = net(X, dec_input, X_valid_len) 

            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()
            # 损失函数的标量进行“反向传播”
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
            f'tokens/sec on {str(device)}')


def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad sequences.
    Defined in :numref:`sec_utils`"""
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad




def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device, save_attention_weights=False):
    """序列到序列模型的预测
    params:
        src_sentence: 数量为batch_size的若干句话
    """

    # 在预测时模型设置为评估模式
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'] ) # shape=(num_steps, )
    # 添加批量轴
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0) # shape=(1, num_steps)
    enc_outputs = net.encoder(enc_X, enc_valid_len) # shape=(1, num_query_enc, num_hiddens)
    print('note1', enc_outputs.shape)

    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len) # 初始化dec_state, 包含encoder部分的 "输出" 和 "valid_len", 以及decoder部分的初始“前序tokens”

    # 添加批量轴
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0) # shape=(1, 1)

    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        # dec_X: shape=(batch_size, 1, num_hiddens), 在masked multihead attention中充当query
        Y, dec_state = net.decoder(dec_X, dec_state) # Y shape=(batch_size, 1, target_vocab_size)
        print('note30', dec_X.shape)
        print('note3', Y.shape)

        # 使用有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2) # dec_X shape=(batch_size, 1)
        print('note4', dec_X.shape)

        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # print('note5', pred.shape)

        # 保存注意力权重（稍后讨论）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq































        
