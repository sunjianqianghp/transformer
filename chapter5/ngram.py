from ..utils.util import Vocab
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from torch import optim
from torch.utils.data import Dataset, DataLoader
from .constants import PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, DEVICE
from .utils import save_pretrained, load_reuters, get_loader


# 从Dataset类（在torch.utils.data中定义）中派生出一个子类
class NGramDataset(Dataset):
    def __init__(self, corpus, vocab, context_size=2):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        for sentence in tqdm(corpus, desc="Dataset Construction"):
            # 插入句首句尾符号
            sentence = [self.bos] + sentence + [self.eos]
            if len(sentence) < context_size:
                continue
            for i in range(context_size, len(sentence)):
                # 模型输入：长为context_size的上文
                context = sentence[i-context_size:i]
                # 模型输出：当前词
                target = sentence[i]
                self.data.append((context, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate_fn(self, examples):
        # 从独立样本集合中构建batch输入输出
        inputs  = torch.tensor([ex[0] for ex in examples], dtype=torch.long)
        targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
        return (inputs, targets)

WEIGHT_INIT_RANGE = 0.1

def init_weights(model):
    for name, param in model.named_parameters():
        # print('name', name)
        if "embedding" not in name:
            torch.nn.init.uniform_(
                param, a=-WEIGHT_INIT_RANGE, b=WEIGHT_INIT_RANGE
            )

class FeedForwardNNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim):
        super(FeedForwardNNLM, self).__init__()

        # 词嵌入层
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # 线性变换：词嵌入层->隐含层
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        
        # 线性变换：隐含层->输出层
        self.linear2 = nn.Linear(hidden_dim, vocab_size)
        
        # 使用ReLU激活函数
        self.activate = F.relu
        init_weights(self)

    def forward(self, inputs):
        '''
        params:
            inputs: shape=(batch_size, context_size)
        '''
        embeds = self.embeddings(inputs).view((inputs.shape[0], -1)) # shape = (batch_size, context_size*embedding_dim)
        hidden = self.activate(self.linear1(embeds)) # shape = (batch_size, hidden_dim)
        output = self.linear2(hidden) # shape = (batch_size, vocab_size)

        # 根据输出层（logits）计算概率分布并取对数，以便于计算对数似然
        # 这里采用PyTorch库的log_softmax实现
        log_probs = F.log_softmax(output, dim=1) # shape = (batch_size, vocab_size)
        return log_probs


if __name__ == "__main__":

    embedding_dim = 128
    hidden_dim = 256
    batch_size = 1024
    context_size = 3
    num_epoch = 1 

    # 读取文本数据，构建FFNNLM训练数据集（n-grams）
    corpus, vocab = load_reuters()
    dataset = NGramDataset(corpus, vocab, context_size)
    print('daflaskdfj', len(dataset))
    data_loader = get_loader(dataset, batch_size)

    # 负对数似然损失函数
    nll_loss = nn.NLLLoss()

    # 构建FFNNLM，并加载至device
    model  = FeedForwardNNLM(len(vocab), embedding_dim, context_size, hidden_dim)
    model.to(DEVICE)

    # 使用Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    total_losses = []
    for epoch in range(num_epoch):
        total_loss = 0
        for batch in tqdm(data_loader, desc=f"Training Epoch {epoch}"):
            inputs, targets = [x.to(DEVICE) for x in batch]
            optimizer.zero_grad()
            log_probs = model(inputs) # shape = (batch_size, vocab_size)
            loss = nll_loss(log_probs, targets) # shape = (batch_size,)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Loss: {total_loss:.2f}")
        total_losses.append(total_loss)

    # 保存词向量（model.embeddings）
    save_pretrained(vocab, model.embeddings.weight.data, "ffnnlm.vec")

