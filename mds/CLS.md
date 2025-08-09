在 Transformer 架构（如 BERT）中，`[CLS]` token 是一个非常重要的**特殊标记（token）**，用于表示整段输入序列的整体信息，尤其在分类任务中扮演关键角色。

---

### 🔹 什么是 `[CLS]` token？

* `CLS` 是 “**classification**” 的缩写。

* 它是一个 **虚拟的占位 token**，在输入文本序列的 **最前面添加**，例如：

  ```
  输入句子（BERT）：
  [CLS] 我 喜欢 机器学习 [SEP]
  ```

* 在 BERT 中，模型会对整个序列进行编码，其中 `[CLS]` 的最终隐藏状态（即输出向量）被用作 **整段输入的语义表示**。

---

### 🔹 `[CLS]` 的作用是什么？

| 任务类型       | `[CLS]` 的用途                        |
| ---------- | ---------------------------------- |
| 文本分类任务     | 用 `[CLS]` 的输出向量作为整个句子的表示，接入分类器输出标签 |
| 文本对匹配任务    | 例如自然语言推理（NLI）， `[CLS]` 表示两个句子的综合信息 |
| 情感分析、主题分类等 | 都是用 `[CLS]` 向量做为句子的“全局摘要”进行判断      |

---

### 🔹 示例：BERT 文本分类

```python
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("hello world", return_tensors="pt")
outputs = model(**inputs)

cls_embedding = outputs.last_hidden_state[:, 0, :]  # 第0个位置就是 [CLS]
```

---

### 🔹 与其他 token 的区别

| 特殊 token | 作用              |
| -------- | --------------- |
| `[CLS]`  | 分类任务的全局表示       |
| `[SEP]`  | 用于分隔句子或段落       |
| `[PAD]`  | 用于补齐序列到相同长度     |
| `[MASK]` | 用于掩盖词预测（MLM 任务） |

---

### 🔹 总结

* `[CLS]` 是 Transformer 模型中用于整段输入**全局表示**的特殊 token。
* 它主要用于 **分类任务** 的输入处理和特征提取。
* 你可以将 `[CLS]` 向量理解为模型对输入序列的“**摘要特征**”。

如果你想知道 `[CLS]` 向量在实际下游任务中是怎么用的，比如接全连接层、多任务微调等，我可以继续深入讲解。
