from utils.util import Vocab, torch, pad_sequence, pack_padded_sequence, pad_packed_sequence, F, BowDataset
from torch.utils.data import DataLoader
from torch import nn, optim


def load_treebank():
    from nltk.corpus import treebank 
    # sents存储全部经过标记化的句子
    # postags存储每个标记对应的词性标注结果
    # sents, postags = zip()

    sents, postags = zip(*(zip(*sent) for sent in treebank.tagged_sents() )) 

    # "<pad>"为预留的用于补齐序列长度的标记
    vocab:Vocab = Vocab.build(sents, reserved_token=["<pad>"])

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

        

if __name__ == '__main__':
    # tqdm是一个Python模块，能以进度条的方式显示迭代的进度
    from tqdm.auto import tqdm

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 超参数设置
    embedding_dim = 128
    hidden_dim =  256
    num_class = 2
    batch_size = 32
    num_epoch = 5

    # 加载数据
    train_data, test_data, vocab, tag_vocab = load_treebank()
    train_dataset = BowDataset(train_data)
    test_dataset  = BowDataset(test_data)

    # --------------------------------------------------  特殊 start  ----------------------------------------------------------
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True )
    test_data_loader  = DataLoader( test_dataset, batch_size=1,          collate_fn=collate_fn, shuffle=False)

    # 加载模型
    model = LSTM(len(vocab), embedding_dim, hidden_dim, num_class=len(tag_vocab))
    # model = Transformer(len(vocab), embedding_dim, hidden_dim, num_class=num_class, 
    #                     dim_feedforward=512, num_head=2, num_layers=2, dropout=0.1, 
    #                     max_len=512, activation='relu')
    # --------------------------------------------------  特殊 end  ----------------------------------------------------------



    model.to(device) # 将模型加载到CPU或GPU设备

    #训练过程
    nll_loss = nn.NLLLoss() # Negative Log Likelihood, NLL
    optimizer = optim.Adam(model.parameters(), lr=0.001) # 使用Adam优化器

    model.train()
    for epoch in range(num_epoch):
        total_loss = 0
        acc = 0
        total = 0
        for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch}"):

            # --------------------------------------------------  特殊 start  ----------------------------------------------------------
            inputs, lengths, targets, mask = [x.to(device) for x in batch]

            # print(inputs.shape, lengths.shape, targets.shape, mask.shape)

            log_probs = model(inputs, lengths)
            # --------------------------------------------------  特殊 end  ----------------------------------------------------------
            
            loss = nll_loss(log_probs[mask], targets[mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc += (log_probs.argmax(dim=-1) == targets)[mask].sum().item()
            total += mask.sum().item()
            total_loss += loss.item()

        print(f"Loss: {total_loss:.2f}, accuracy: {acc/total: .2f}")

    # 测试过程
    
    total_loss = 0
    acc = 0
    total = 0
    model.eval()
    for batch in tqdm(test_data_loader, desc=f"Testing"):
        # --------------------------------------------------  特殊 start  ----------------------------------------------------------
        inputs, lengths, targets, mask = [x.to(device) for x in batch]
        # --------------------------------------------------  特殊 end  ----------------------------------------------------------
        with torch.no_grad():
            # --------------------------------------------------  特殊 start  ----------------------------------------------------------
            log_probs = model(inputs, lengths)
            # --------------------------------------------------  特殊 end  ----------------------------------------------------------
            
            loss = nll_loss(log_probs[mask], targets[mask])

            acc += (log_probs.argmax(dim=-1) == targets)[mask].sum().item()
            total += mask.sum().item()
            total_loss += loss.item()

    # 输出在测试集上的准确率
    print(f"Loss: {total_loss:.2f}, accuracy: {acc/total: .2f}")
