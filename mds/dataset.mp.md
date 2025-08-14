`DatasetDict.map()` 是 Hugging Face **datasets** 库里用来**批量处理数据**的一个核心方法，它可以在 **`DatasetDict`**（包含多个数据集 split，比如 train/validation/test）上批量应用一个函数，返回处理后的新 `DatasetDict`。

我给你详细拆开讲一下。

---

## **1. 基本作用**

* 对数据集的 **每一条数据** 或 **一个 batch** 应用自定义处理函数
* 常用于：

  * 文本预处理（分词、清理、拼接等）
  * 特征工程（计算额外字段）
  * 标签转换
  * 数据增强

---

## **2. 基本语法**

```python
dataset_dict = dataset_dict.map(
    function,               # 处理函数
    batched=False,          # 是否批处理
    batch_size=1000,        # batched=True 时批大小
    remove_columns=None,    # 删除指定列
    keep_in_memory=False,   # 是否全部加载到内存
    load_from_cache_file=True, # 是否使用缓存
    num_proc=None           # 并行进程数
)
```

---

## **3. 参数说明**

| 参数                          | 类型             | 作用                                         |
| --------------------------- | -------------- | ------------------------------------------ |
| **function**                | `callable`     | 用来处理数据的函数，输入是字典或批量字典，输出也是字典                |
| **batched**                 | `bool`         | 是否批量传入数据，如果 `True`，`function` 接收到的是包含列表的字典 |
| **batch\_size**             | `int`          | 批大小，仅在 `batched=True` 时有效                  |
| **remove\_columns**         | `str` / `list` | 处理后删除的列名                                   |
| **keep\_in\_memory**        | `bool`         | 是否将结果保存在内存中（避免频繁读写磁盘）                      |
| **load\_from\_cache\_file** | `bool`         | 是否从缓存加载结果，`False` 可强制重新计算                  |
| **num\_proc**               | `int`          | 并行进程数（加快处理速度）                              |

---

## **4. 示例**

### **① 对所有 split 分词**

```python
from datasets import load_dataset
from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
dataset = load_dataset("glue", "rte")  # DatasetDict(train, validation, test)

def tokenize_fn(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

dataset_tokenized = dataset.map(tokenize_fn, batched=True)
```

---

### **② 删除原始文本列**

```python
dataset_tokenized = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=["sentence1", "sentence2"]
)
```

---

### **③ 并行加速处理**

```python
dataset_tokenized = dataset.map(tokenize_fn, batched=True, num_proc=4)
```

---

### **④ 修改标签**

```python
def label_map(example):
    example["label"] = int(example["label"]) * 10
    return example

dataset_new = dataset.map(label_map)
```

---

## **5. DatasetDict.map() 与 Dataset.map() 区别**

* **`DatasetDict.map()`**：会自动在字典中每个 split（train、validation、test）上运行
* **`Dataset.map()`**：只对单个 split 运行

例：

```python
dataset["train"].map(...)       # 只处理 train
dataset.map(...)                # 自动处理 train/validation/test 全部
```

---
