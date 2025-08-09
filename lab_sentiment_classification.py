import math
import pandas as pd
import torch 
from torch import nn, optim
from torch.utils.data import DataLoader
from utils.util import *


if __name__ == "__main__":
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
    train_data, test_data, vocab = load_sentence_polarity()
    train_dataset = BowDataset(train_data)
    test_dataset  = BowDataset(test_data)

    # --------------------------------------------------  特殊 start  ----------------------------------------------------------
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn_lstm, shuffle=True )
    test_data_loader  = DataLoader( test_dataset, batch_size=1,          collate_fn=collate_fn_lstm, shuffle=False)

    # 加载模型
    # model = MLP(len(vocab), embedding_dim, hidden_dim, num_class)
    # model = CNN(len(vocab), embedding_dim, filter_size=3, num_filter=100, num_class=num_class)
    model = LSTM(len(vocab), embedding_dim, hidden_dim, num_class=num_class)
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
        for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch}"):

            # --------------------------------------------------  特殊 start  ----------------------------------------------------------
            # inputs, offsets, targets = [x.to(device) for x in batch]
            # inputs, targets = [x.to(device) for x in batch]
            inputs, lengths, targets = [x.to(device) for x in batch]

            # print(inputs.shape)

            # log_probs = model(inputs, offsets)
            # log_probs = model(inputs)
            log_probs = model(inputs, lengths)
            # --------------------------------------------------  特殊 end  ----------------------------------------------------------
            
            loss = nll_loss(log_probs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Loss: {total_loss:.2f}")

    # 测试过程
    
    acc = 0
    model.eval()
    for batch in tqdm(test_data_loader, desc=f"Testing"):
        # --------------------------------------------------  特殊 start  ----------------------------------------------------------
        # inputs, offsets, targets = [x.to(device) for x in batch]
        inputs, lengths, targets = [x.to(device) for x in batch]
        # --------------------------------------------------  特殊 end  ----------------------------------------------------------
        with torch.no_grad():
            # --------------------------------------------------  特殊 start  ----------------------------------------------------------
            # output = model(inputs, offsets)
            # output = model(inputs)
            output = model(inputs, lengths)
            # --------------------------------------------------  特殊 end  ----------------------------------------------------------
            acc += (output.argmax(dim=1) == targets).sum().item()

    # 输出在测试集上的准确率
    print(f"Acc: {acc / len(test_data_loader):.2f}")




    






