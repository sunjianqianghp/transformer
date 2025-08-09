import torch
from torch.utils.data import Dataset, DataLoader 
from .constants import PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, DEVICE
from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence 
from torch import nn 
import torch.nn.functional as F
from .utils import save_pretrained, load_reuters, get_loader


class RnnlmDataset(Dataset):
    def __init__(self, corpus, vocab):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        self.pad = vocab[PAD_TOKEN]
        for sentence in tqdm(corpus, desc="Dataset Construction"):
            # 模型输入序列： BOS_TOKEN, w_1, w_2, ..., w_n 
            input = [self.bos] + sentence 
            # 模型输出序列： w_1, w_2, ..., w_n, EOS_TOKEN 
            target = sentence + [self.eos]
            self.data.append((input, target))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]
    
    def collate_fn(self, examples):
        # 从独立样本集合中构建batch输入输出
        inputs  = [ex[0] for ex in examples]
        targets = [ex[1] for ex in examples]
        # 对批次内的样本进行长度补齐 
        inputs  = pad_sequence(inputs, batch_first=True, padding_value=self.pad)
        # 对输出序列进行补齐
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad)
        return (inputs, targets)

class RNNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNLM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 循环神经网络，这里使用LSTM
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # 输出层
        self.output = nn.Linear(hidden_dim, vocab_size) 

    def forward(self, inputs):
        '''
        inputs: (batch_size, seq_len)
        '''
        embeds = self.embeddings(inputs) # shape: (batch_size, seq_len, embedding_dim)
        # 计算每一时刻的隐含层表示
        hidden, _ = self.rnn(embeds) # shape: (batch_size, seq_len, hidden_dim)
        output = self.output(hidden) # shape: (batch_size, seq_len, vocab_size)
        log_probs = F.log_softmax(output, dim=2) # shape: (batch_size, seq_len, vocab_size)
        return log_probs
    

if __name__ == "__main__":
    num_epochs = 10


    corpus, vocab = load_reuters()
    dataset = RnnlmDataset(corpus, vocab)
    data_loader = get_loader(dataset, batch_size=32, shuffle=True)

    # 负对数似然损失函数，设置ignore_index参数，以忽略PAD_TOKEN的损失
    nll_loss = nn.NLLLoss(ignore_index=dataset.pad)
    # 构建RNNLM， 并加载至相应设备 
    model = RNNLM(len(vocab), embedding_dim=100, hidden_dim=256)
    model.to(DEVICE)

    # 使用Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(data_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            inputs, targets = [x.to(DEVICE) for x in batch] # inputs: (batch_size, seq_len), targets: (batch_size, seq_len)
            optimizer.zero_grad()
            log_probs = model(inputs) # shape: (batch_size, seq_len, vocab_size)
            # 计算负对数似然损失
            loss = nll_loss(log_probs.view(-1, log_probs.shape[-1]), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()        
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}")

# 将词向量保存至rnnlm.vec文件
save_pretrained(vocab, model.embeddings.weight.data, "rnnlm.vec")
