`Dataset.set_format` 是 Hugging Face **datasets** 库里用来**设置数据集返回格式**的方法，它会影响你用 `dataset[i]` 或 `dataset[:n]` 访问数据时的输出类型。

---

## **作用**

* 控制数据返回的**数据类型**（如 `numpy`、`torch`、`tensorflow`、`pandas` 等）
* 指定要返回的**列**（column）
* 节省内存和提升性能（因为只会加载你需要的列，并且直接转成需要的类型）

---

## **基本语法**

```python
dataset.set_format(
    type=None,       # 输出类型: 'numpy', 'torch', 'tensorflow', 'pandas', 或 None
    columns=None,    # 要保留的列列表（默认保留所有列）
    output_all_columns=False
)
```

---

## **参数说明**

| 参数                   | 说明                                                                                                                                            |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `type`               | 数据输出类型。常见取值：`'numpy'`（返回 np.array）、`'torch'`（返回 torch.Tensor）、`'tensorflow'`（返回 tf.Tensor）、`'pandas'`（返回 pd.DataFrame）、`None`（返回 Python 原生类型） |
| `columns`            | 要包含在输出中的列名列表。其他列会被隐藏（但仍在磁盘上，不会丢失）。                                                                                                            |
| `output_all_columns` | 如果为 `True`，即使指定了 `columns`，也会返回所有列（只是 `columns` 中的列会被转换类型）。                                                                                   |

---

## **使用示例**

### 1. 返回 PyTorch 张量

```python
from datasets import load_dataset

dataset = load_dataset("glue", "mrpc", split="train")

# 只保留输入文本和标签，并转换为 torch.Tensor
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

print(dataset[0])  # 直接得到 torch.Tensor
```

---

### 2. 返回 NumPy 数组

```python
dataset.set_format(type="numpy", columns=["label"])
print(dataset[:5]["label"])  # 输出 np.ndarray
```

---

### 3. 恢复默认格式

```python
dataset.reset_format()
```

---

✅ **小技巧**：

* 在训练模型前，用 `set_format(type="torch", columns=[...])` 可以省去手动 `torch.tensor()` 的步骤。
* 如果你只是做数据分析，`set_format(type="pandas")` 可以直接用 `dataset[:].to_pandas()` 分析数据。

---

如果你需要，我可以帮你画一个\*\*`set_format` 数据流转换图\*\*，形象展示数据从磁盘到张量的流程，这样你会更直观地理解它的作用。你需要吗？
