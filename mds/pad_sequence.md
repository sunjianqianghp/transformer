`torch.nn.utils.rnn.pad_sequence` 是 PyTorch 中一个非常实用的工具函数，主要用于将**不同长度的序列**补齐（padding）为统一长度，方便批处理。

这个函数在处理 **变长序列（variable-length sequences）** 时非常常用，尤其是在 NLP、序列建模、RNN 输入等场景中。

---

## 🧩 函数作用

**将多个不同长度的序列补齐为相同长度的张量（Tensor），以便可以组成一个 batch。**

---

## 🔧 语法

```python
torch.nn.utils.rnn.pad_sequence(sequences, batch_first=False, padding_value=0.0)
```

---

## ✅ 参数说明

| 参数              | 说明                                                        |
| --------------- | --------------------------------------------------------- |
| `sequences`     | 一个包含多个 1D 或 2D 张量的 list 或 tuple，每个张量代表一个序列，长度可变           |
| `batch_first`   | 是否将 batch 维放在第一个维度（`True` 为 `[batch_size, seq_len, ...]`） |
| `padding_value` | 用于填充的值（默认是 0.0）                                           |

---

## 📌 示例

### 🔤 举例：处理变长句子

```python
import torch
from torch.nn.utils.rnn import pad_sequence

# 假设有三个不同长度的序列（每个是一个 Tensor）
seq1 = torch.tensor([1, 2, 3])
seq2 = torch.tensor([4, 5])
seq3 = torch.tensor([6])

# 放在 list 中
sequences = [seq1, seq2, seq3]

# 进行 padding（默认是 seq_len 在第一维）
padded = pad_sequence(sequences, batch_first=True, padding_value=0)

print(padded)
```

### 🔢 输出结果：

```python
tensor([[1, 2, 3],
        [4, 5, 0],
        [6, 0, 0]])
```

* 每行是一个序列，已经补齐到等长。
* 填充位置使用的是 `0`（默认或自定义的 padding\_value）。

---

## 📐 维度说明

| 参数设置                | 输出维度                        |
| ------------------- | --------------------------- |
| `batch_first=True`  | `[batch_size, max_seq_len]` |
| `batch_first=False` | `[max_seq_len, batch_size]` |

---

## 🚀 通常配合使用场景

### 配合以下模块使用：

* `torch.nn.Embedding`（进行词向量映射）
* `torch.nn.utils.rnn.pack_padded_sequence`（RNN 中进一步压缩 padding）
* `torch.nn.LSTM` / `GRU`（在处理变长序列时）

---

## 🔄 示例：配合 LSTM 使用

```python
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

# 假设有嵌入后的变长序列
embedded = [torch.randn(l, 10) for l in [5, 3, 2]]  # 3个句子，长度分别为5, 3, 2

# padding
padded = pad_sequence(embedded, batch_first=True)  # shape: [3, 5, 10]

# 序列原始长度（必要）
lengths = torch.tensor([5, 3, 2])

# pack 成 PackedSequence 对象（喂给 LSTM）
packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
```

---

## ✅ 总结

| 优点        | 说明                                           |
| --------- | -------------------------------------------- |
| 🧩 对齐序列   | 把不同长度的序列补齐成统一长度，方便批量处理                       |
| 🔗 兼容RNN  | 常与 `pack_padded_sequence` 连用，以支持变长序列         |
| ⚙️ 自定义填充值 | 可自定义 `padding_value`，适配不同场景（如 `<PAD>` token） |
| 📐 灵活维度   | 支持 `batch_first` 控制维度顺序                      |

---

如果你想我再演示一段完整的 NLP 示例（例如处理一批句子文本），也可以告诉我，我可以从 `tokenize` 到 `Embedding + LSTM` 一步步帮你展示。是否需要？
