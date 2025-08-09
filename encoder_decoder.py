import math
import torch
from torch import nn
from components import MultiHeadAttention, AddNorm, PositionWiseFFN, PositionalEncoding, train_seq2seq

class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        
    def forward(self, X, *args):
        """
        X: 长度可变的序列作为编码器的输入
        """
        raise NotImplementedError
    

class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        """将编码器的输出（enc_outputs）转换为编码后的状态"""
        raise NotImplementedError
    
    def forward(self, X, state):
        raise NotImplementedError
    
#@save
class AttentionDecoder(Decoder):
    """带有注意力机制解码器的基本接口"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError
    
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder 
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)


class EncoderBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, dim_key, dim_query, dim_value, num_hiddens, 
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, 
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.mh_attention = MultiHeadAttention(
            dim_key, dim_query, dim_value, num_hiddens, num_heads, dropout, use_bias
        )
        print('encoder', norm_shape)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        # print(f'encoder self.training {self.training}')
        # print('encoder attention', X.shape, valid_lens.shape)
        Y = self.addnorm1(X, self.mh_attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))
    

class TransformerEncoder(Encoder):
    """Transformer编码器"""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"block{i}",
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))
    
    def forward(self, X, valid_lens, *args):
        """
        因为位置编码值在-1和1之间, 因此嵌入值乘以嵌入维度的平方根进行缩放，然后再与位置编码相加
        params:
            X: shape = (batch_size, num_queries)
        """
        X = self.pos_encoding(self.embedding(X)*math.sqrt(self.num_hiddens)) # shape = (batch_size, num_queries, num_hiddens)
        self.attention_weights = [None]*len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.mh_attention.dp_attention.attention_weights
        return X # shape = (batch_size, num_queries, num_hiddens)
        

class DecoderBlock(nn.Module):
    """解码器第i个块"""
    def __init__(self, dim_key, dim_query, dim_value, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i  # 第i层的解码块
        self.attention1 = MultiHeadAttention(dim_key, dim_query, dim_value, num_hiddens, num_heads, dropout)
        self.attention2 = MultiHeadAttention(dim_key, dim_query, dim_value, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.addnorm3 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)

    def forward(self, X, state):
        """
        params:
            X: shape=(batch_size, num_steps(train) or 1(eval), num_hiddens), 在masked multihead attention中充当queries
            state: [enc_outputs, enc_valid_lens, [key_values#1, key_values#2, ..., key_values#(i-1), None, ...] ],
            state的前两个元素固定不变, 在各层解码块都相同, 第三各元素是长度为num_layers的列表, 本层解码块(第i层)更新第i个元素
        """
        # print(f'decoder self.training {self.training}')
        enc_outputs, enc_valid_lens = state[0], state[1] # enc_outputs充当key和value的作用
        """
        训练阶段, 输出序列的所有词元都在同一时间处理, 因此state[2][self.i]初始化为None.
        预测阶段, 输出序列是通过次元一个接一个解码的, 因此state[2][self.i]包含着直到当前时间步第i个解码的输出表示
        """
        if state[2][self.i] is None:
            key_values = X 
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values

        if self.training:
            batch_size, num_steps, _ = X.shape

            # dec_valid_lens的开头：（batch_size, num_steps), 其中每一行是[1, 2, ..., num_steps]
            dec_valid_lens = torch.arange(1, num_steps+1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        # 自注意力
        # print('attention1', X.shape, key_values.shape, dec_valid_lens.shape)
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens) # shape=(batch_size, num_queries, num_hiddens)(training) or shape=(batch_size, 1, num_hiddens)(eval)
        print('note2', X.shape, X2.shape)
        Y = self.addnorm1(X, X2)
        # 编码器-解码器注意力。
        # enc_outputs的开头：(batch_size, query_size, num_hiddens)
        # print('attention2', Y.shape, enc_outputs.shape, enc_valid_lens.shape)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens) # shape=(batch_size, num_queries, num_hiddens)(training) or shape=(batch_size, 1, num_hiddens)(eval)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state


class TransformerDecoder(AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"block{i}", 
                DecoderBlock(key_size, query_size, value_size, num_hiddens, 
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)
    
    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None]*self.num_layers]
    
    def forward(self, X, state):
        
        X = self.pos_encoding(self.embedding(X)*math.sqrt(self.num_hiddens))
        self._attention_weights = [[None]*len(self.blks) for i in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][i] = blk.attention1.dp_attention.attention_weights
            # “编码器-解码器”自注意力权重
            self._attention_weights[1][i] = blk.attention2.dp_attention.attention_weights
        return self.dense(X), state
    
    @property
    def attention_weights(self):
        return self._attention_weights



if __name__ == '__main__':
    from d2l import torch as d2l
    # x = torch.ones((2, 13, 24))
    # y = torch.ones((2, 17, 24))

    # valid_lens_x = torch.tensor([9, 10])
    # valid_lens_y = torch.tensor([7, 12])
    # encoder_blk = EncoderBlock(24, 24, 24, 24, [13, 24], 24, 48, 8, 0.5)
    # encoder_blk.train()

    # decoder_blk = DecoderBlock(24, 24, 24, 24, [17, 24], 24, 48, 8, 0.5, 0)
    # decoder_blk.train()
    # state = [encoder_blk(x, valid_lens_x), valid_lens_y, [None]]
    # res = decoder_blk(y, state)
    # print('decoder output', res[0].shape) # decoder output
    # print('encoder output', res[1][0].shape) # encoder output
    # print('encoder valid_lens', res[1][1])  # encoder valid_lens
    # print('decoder input_seq', res[1][2][0].shape )

    num_hiddens = 32
    num_layers = 2
    dropout = 0.1
    batch_size = 64
    num_steps = 10

    lr = 0.005
    num_epochs = 200
    device = d2l.try_gpu()
    
    ffn_num_input = 32
    ffn_num_hiddens = 64
    num_heads = 4

    key_size = 32
    query_size = 32
    value_size = 32

    norm_shape = [32]

    train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

    encoder = TransformerEncoder(
        len(src_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)
    
    decoder = TransformerDecoder(
        len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)

    net = EncoderDecoder(encoder, decoder)
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    nn.TransformerEncoder()



    
