`evaluate.load()` 是 🤗 Hugging Face **Evaluate** 库中的一个核心函数，主要作用是**加载评估指标（metrics）或评估模块（evaluation modules）**，让你在 NLP、CV 等任务中方便地计算常见指标，比如 `accuracy`、`f1`、`rouge`、`bleu` 等。

---

## 1️⃣ 作用

* 从 **Hugging Face Hub** 或本地加载评估指标
* 统一不同指标的使用方式（`.compute()`）
* 方便在模型训练和评估中计算准确率、F1、ROUGE 等指标
* 支持自定义指标（从本地 Python 脚本加载）

---

## 2️⃣ 基本用法

```python
import evaluate

# 加载一个内置指标，比如准确率
accuracy = evaluate.load("accuracy")

# 模拟预测和标签
y_pred = [0, 1, 1, 0]
y_true = [0, 1, 0, 0]

# 计算指标
results = accuracy.compute(predictions=y_pred, references=y_true)
print(results)
```

输出：

```python
{'accuracy': 0.75}
```

---

## 3️⃣ 参数说明

```python
evaluate.load(path, config_name=None, module_type=None, cache_dir=None, download_mode=None)
```

| 参数                 | 说明                                                                 |
| ------------------ | ------------------------------------------------------------------ |
| **path**           | 评估模块名称（如 `"accuracy"`、`"f1"`、`"bleu"`）或本地路径                        |
| **config\_name**   | 有些指标有多种配置，比如 `"glue"` 数据集的不同任务：`evaluate.load("glue", "rte")`      |
| **module\_type**   | `"metric"` 或 `"comparison"`，一般不用手动设                                |
| **cache\_dir**     | 缓存目录（不指定则默认到 `~/.cache/huggingface/modules/evaluate_metrics/`）     |
| **download\_mode** | 是否强制重新下载，选项：`"reuse_dataset_if_exists"`（默认）、`"force_redownload"` 等 |

---

## 4️⃣ 典型用例

### (1) 加载多个指标

```python
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")

preds = [0, 1, 1, 0]
labels = [0, 1, 0, 0]

print(f1_metric.compute(predictions=preds, references=labels, average="macro"))
print(precision_metric.compute(predictions=preds, references=labels, average="macro"))
```

---

### (2) 评估 GLUE 中的某个任务

```python
glue_metric = evaluate.load("glue", "rte")
preds = [0, 1, 1, 0]
labels = [0, 1, 0, 0]
print(glue_metric.compute(predictions=preds, references=labels))
```

---

### (3) 从本地 Python 脚本加载自定义指标

```python
custom_metric = evaluate.load("./my_metrics/my_accuracy.py")
result = custom_metric.compute(predictions=[1,0], references=[1,1])
print(result)
```

其中 `my_accuracy.py` 必须定义 `compute()` 函数返回结果字典。

---

## 5️⃣ 注意事项

* `evaluate` 是独立于 `datasets` 的库（但语法风格相似）
* 如果无法下载指标（网络问题），可以手动下载到本地，然后用 `evaluate.load(local_path)` 加载
* 一些任务型指标（如 `glue`、`squad`）需要传 `config_name`

---

如果你需要的话，我可以帮你整理一份 **evaluate 常用指标速查表**，这样你不用每次去查 Hugging Face 文档就能直接调用。
你需要我帮你整理吗？这样会更方便你直接用。
