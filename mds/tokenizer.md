我来帮你详细讲一下 **Hugging Face `tokenizer`** 的主要参数及作用，并用 BERT 为例说明。
这里的 `tokenizer` 指的是 `BertTokenizer` / `BertTokenizerFast` 以及其它模型对应的分词器，它们的参数基本通用。

---

## **1. 基本用法**

```python
from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
encoded = tokenizer(
    text="I love NLP.",
    text_pair="It is amazing.",
    max_length=16,
    padding="max_length",
    truncation=True,
    return_tensors="pt"
)
print(encoded)
```

---

## **2. 常用参数说明**

| 参数                                | 类型                                        | 作用                                    | 示例值                                                |
| --------------------------------- | ----------------------------------------- | ------------------------------------- | -------------------------------------------------- |
| **text / text\_pair**             | `str` / `List[str]`                       | 输入文本，可传一个句子或句对                        | `"I love NLP"` / `("I love NLP", "It is amazing")` |
| **max\_length**                   | `int`                                     | 最大序列长度（超过则截断）                         | `max_length=128`                                   |
| **padding**                       | `bool` / `"max_length"` / `"longest"`     | 是否补齐到相同长度                             | `"max_length"` 补到 `max_length`，`True` 补到 batch 中最长 |
| **truncation**                    | `bool` / `"only_first"` / `"only_second"` | 是否截断超长文本                              | `True` 截断所有超长句                                     |
| **return\_tensors**               | `"pt"` / `"tf"` / `"np"`                  | 返回的张量类型（PyTorch / TensorFlow / NumPy） | `"pt"`                                             |
| **return\_token\_type\_ids**      | `bool`                                    | 是否返回 `token_type_ids`（句子区分）           | `True`（默认）                                         |
| **return\_attention\_mask**       | `bool`                                    | 是否返回 `attention_mask`（有效 token 标记）    | `True`（默认）                                         |
| **add\_special\_tokens**          | `bool`                                    | 是否添加特殊符号（如 `[CLS]`、`[SEP]`）           | `True`（默认）                                         |
| **stride**                        | `int`                                     | 当 `truncation` 启用时，用于滑动窗口的重叠 token 数  | `stride=2`                                         |
| **return\_offsets\_mapping**      | `bool`                                    | 返回原文本与 token 的字符位置映射（NER 等任务用）        | `True`                                             |
| **is\_split\_into\_words**        | `bool`                                    | 输入是否已是分好词的列表                          | `True` 表示跳过分词                                      |
| **return\_special\_tokens\_mask** | `bool`                                    | 返回哪些 token 是特殊符号的 mask                | `True`                                             |
| **verbose**                       | `bool`                                    | 是否显示警告信息                              | `False`                                            |

---

## **3. 输出内容**

常见输出包含：

* **`input_ids`**：分词后对应的 token ID 序列
* **`token_type_ids`**：区分句子 A/B
* **`attention_mask`**：标记哪些 token 有效
* **`offset_mapping`**：原文字符位置到 token 的映射（如果开启）

示例：

```python
{
 'input_ids': tensor([[ 101, 146, 1567, 17953, 119, 102, 1135, 1110, 5941, 119, 102, 0, 0, 0, 0, 0]]),
 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]]),
 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
}
```

---

## **4. 常见组合写法**

```python
# 单句
tokenizer("Hello world", padding="max_length", truncation=True, max_length=8)

# 句对
tokenizer("Hello world", "How are you?", padding=True)

# 批量输入
tokenizer(["Hello world", "How are you?"], padding=True, return_tensors="pt")

# 已分好词
tokenizer([["Hello", "world"], ["How", "are", "you"]], is_split_into_words=True, padding=True)
```

---

如果你愿意，我可以帮你画一张 **tokenizer 参数作用示意图**，把 `input_ids`、`token_type_ids`、`attention_mask`、`padding`、`truncation` 等的效果全可视化，这样更直观。
你要我画吗？
