`torch.nn.Embedding` 和 `torch.nn.EmbeddingBag` 都是用于词嵌入（embedding）的 PyTorch 模块，它们都能把**离散的整数索引**映射为**连续的向量表示**，但使用方式和应用场景有所不同。

下面是详细的对比：

---

## 🧩 1. `torch.nn.Embedding`：基础嵌入层

### ✅ 用途：

* 给定一个词（或索引），查找它对应的嵌入向量。
* 常用于 NLP 中的词表示（如 Word2Vec、GloVe 初始化等）。

### ✅ 典型输入输出：

```python
import torch
import torch.nn as nn

embedding = nn.Embedding(num_embeddings=10, embedding_dim=3)
input = torch.tensor([1, 2, 4, 5])  # 一批 token ids
output = embedding(input)
print(output.shape)  # => torch.Size([4, 3])
```

* 输入：一个或多个 token 的索引（可以是任意形状）
* 输出：对应的嵌入向量，形状为 `[*input.shape, embedding_dim]`

### 📌 特点：

* **逐 token 查找**，不做任何聚合。
* 可以用于序列建模（如 RNN、Transformer）中，保留位置顺序。

---

## 🧮 2. `torch.nn.EmbeddingBag`：加权嵌入聚合层

### ✅ 用途：

* 用于**将一组词的嵌入向量聚合为一个向量**（平均、求和或最大值）。
* 常用于文本分类、推荐系统等任务中，尤其是**Bag-of-Words 或句子级别任务**。

### ✅ 聚合方式：

通过参数 `mode` 控制：

* `'mean'`：平均（默认）
* `'sum'`：求和
* `'max'`：取最大值

### ✅ 典型输入输出：

```python
embedding_bag = nn.EmbeddingBag(num_embeddings=10, embedding_dim=3, mode='mean')

input = torch.tensor([1, 2, 4, 5])         # 共4个token
offsets = torch.tensor([0, 2])             # 表示2个句子： [1,2], [4,5]

output = embedding_bag(input, offsets)
print(output.shape)  # => torch.Size([2, 3])
```

### 📌 特点：

* 会自动对每组词做聚合操作。
* **高效**，避免显式调用 `.mean()` 或 `.sum()`。
* 常用于 NLP 中不需要顺序信息的任务，如 fastText 或文档分类。

---

## 🔍 总结对比表

| 特性              | `nn.Embedding` | `nn.EmbeddingBag`      |
| --------------- | -------------- | ---------------------- |
| 输出              | 每个词的嵌入向量       | 一组词的聚合向量（如整句话）         |
| 是否保留词顺序         | ✅ 是            | ❌ 否                    |
| 是否需要 offsets 参数 | ❌ 不需要          | ✅ 需要（告诉它每组的起始位置）       |
| 是否自动聚合          | ❌ 不聚合          | ✅ 自动聚合（mean/sum/max）   |
| 应用场景            | 语言模型、序列建模      | 文本分类、推荐系统、bag-of-words |
| 效率              | 一般             | 高效（可并行处理多个样本）          |

---

## 🧠 举个例子来区分

假设你有两个句子：

* 句子 1：`[1, 2]`
* 句子 2：`[4, 5]`

使用 `nn.Embedding`：

```python
# 每个token返回一个向量，保留顺序
embedding(torch.tensor([1, 2, 4, 5]))
# 输出 shape: [4, embedding_dim]
```

使用 `nn.EmbeddingBag` + `offsets`：

```python
# 聚合为两个向量（句子级别表示）
embedding_bag(torch.tensor([1, 2, 4, 5]), offsets=torch.tensor([0, 2]))
# 输出 shape: [2, embedding_dim]
```

---

如果你告诉我具体应用（比如：文本分类、语言建模、图神经网络等），我可以进一步帮你选择哪个更合适，甚至写一个完整 demo。需要吗？
