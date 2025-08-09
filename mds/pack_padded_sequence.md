`torch.nn.utils.rnn.pack_padded_sequence` 是 PyTorch 中处理 **可变长度序列** 的一个非常重要的工具，常用于将不等长序列输入到 **RNN**（如 LSTM、GRU）中。

---

## 🧠 作用概述

`pack_padded_sequence` 将一个 **按长度排序并填充过的 batch 序列**（`padded sequence`），打包为一个紧凑的数据结构，供 RNN 使用，以提高计算效率并避免对 padding 的无效计算。

---

## 🧩 适用场景

你有一个批量的序列数据，它们长度不一，比如：

```python
sequences = [
    [1, 2, 3],
    [4, 5],
    [6]
]
```

为了放进神经网络中，必须将它们补齐（padding）为相同长度：

```python
padded = [
    [1, 2, 3],
    [4, 5, 0],
    [6, 0, 0]
]
```

但你不希望 RNN 处理 padding 的 0，于是你使用 `pack_padded_sequence`。

---

## ✅ 函数原型

```python
torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True)
```

### 参数说明：

* **`input`**: 填充后的 Tensor，形状为 `(seq_len, batch, input_size)` 或 `(batch, seq_len, input_size)`（取决于 `batch_first`）
* **`lengths`**: 一个 1D tensor 或 list，表示每个序列的真实长度（不包括 padding 部分）
* **`batch_first`**: 如果为 True，则 `input` 的形状是 `(batch, seq_len, input_size)`
* **`enforce_sorted`**: 如果为 True，`lengths` 必须按降序排列。如果为 False，会在内部进行排序。

---

## 🔁 使用示例

```python
import torch
from torch.nn.utils.rnn import pack_padded_sequence

# 三个序列：长度分别为 3, 2, 1
inputs = torch.tensor([
    [[1], [2], [3]],
    [[4], [5], [0]],
    [[6], [0], [0]]
], dtype=torch.float)  # (batch, seq_len, input_size)

lengths = torch.tensor([3, 2, 1])

# 打包
packed = pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)
```

---

## 📦 输出结构

输出是一个 `PackedSequence` 对象，它会：

* 储存打包后的有效数据（不含 padding）
* 记录每个时间步的 batch size（用于还原）

你可以直接把这个 `packed` 对象传给 `nn.LSTM` 或 `nn.GRU`。

---

## 🔄 还原序列

使用 `pad_packed_sequence` 来还原：

```python
from torch.nn.utils.rnn import pad_packed_sequence

output, lengths = pad_packed_sequence(packed_output, batch_first=True)
```

---

## ⚠️ 注意事项

1. 若 `enforce_sorted=True`，必须先按 `lengths` 降序排列输入数据；
2. 仅适用于支持可变长度输入的模块（如 RNN/LSTM/GRU）；
3. 不适用于 Transformer，这类模型通常通过 Mask 来处理 padding。

---

## ✅ 总结

| 函数                     | 作用                                  |
| ---------------------- | ----------------------------------- |
| `pack_padded_sequence` | 将填充后的不等长序列打包，去除 padding，便于高效输入到 RNN |
| `pad_packed_sequence`  | 将打包后的序列还原回 padded 形式，便于后处理          |

是否需要我帮你写一个完整示例（包括 LSTM 使用）？
