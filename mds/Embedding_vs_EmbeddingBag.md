`torch.nn.EmbeddingBag` 和 `torch.nn.Embedding` 都是 PyTorch 中用于处理嵌入（embedding）的模块，但它们的用途和行为略有不同。下面详细解释它们的区别：

---

## ✅ 一句话区别：

* **`Embedding`**：返回每个 token 的嵌入向量。
* **`EmbeddingBag`**：返回一组 token 嵌入的聚合（如平均、求和），**节省显存和计算**，特别适合 NLP 中的 bag-of-words 模型。

---

## 🧠 1. `torch.nn.Embedding`

### 功能：

将输入的整数（表示单词/索引）映射为嵌入向量。

### 用法：

```python
import torch
import torch.nn as nn

embedding = nn.Embedding(num_embeddings=10, embedding_dim=3)
input = torch.LongTensor([1, 2, 4, 5])
output = embedding(input)
```

### 输出：

形状为 `[input_size, embedding_dim]`，即每个索引都有一个嵌入向量。

---

## ⚡️ 2. `torch.nn.EmbeddingBag`

### 功能：

和 `Embedding` 类似，但**自动对嵌入结果进行聚合（如 sum/mean/max）**，适用于比如句子或段落中单词的 bag-of-words 表示。

### 用法：

```python
embedding_bag = nn.EmbeddingBag(num_embeddings=10, embedding_dim=3, mode='mean')
input = torch.LongTensor([1, 2, 4, 5])      # 所有的 token 索引
offsets = torch.LongTensor([0, 2])          # 每个“句子”起始索引
output = embedding_bag(input, offsets)
```

### 输出：

每个 offset 表示一个句子的起始位置，输出是 `[num_sentences, embedding_dim]`。自动聚合每个句子的词嵌入。

---

## 📌 总结对比表：

| 特性   | `Embedding`                       | `EmbeddingBag`                |
| ---- | --------------------------------- | ----------------------------- |
| 输入   | Token 索引序列                        | 所有 token 的扁平索引 + 每个句子的 offset |
| 输出   | 每个 token 的嵌入向量                    | 聚合后的句子嵌入向量（每个句子一个）            |
| 聚合操作 | 无                                 | 支持 `mean`、`sum`、`max`         |
| 适合场景 | 每个 token 需要独立处理，如 RNN/Transformer | Bag-of-Words、文本分类、推荐系统等       |
| 效率   | 普通                                | 更高效（内置聚合操作，节省显存和计算）           |

---

## 🧪 示例：

```python
# Example with EmbeddingBag
embedding_bag = nn.EmbeddingBag(10, 3, mode='mean')
input = torch.LongTensor([1, 2, 4, 5])      # Tokens
offsets = torch.LongTensor([0, 2])          # Two sentences: [1,2], [4,5]
output = embedding_bag(input, offsets)
print(output.shape)  # => torch.Size([2, 3])
```

---

如果你告诉我你在做哪种 NLP 任务（比如分类、序列标注、推荐系统），我可以建议更适合的选择。
