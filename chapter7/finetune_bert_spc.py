# Defined in Section 7.4.3.2
from os.path import join, dirname
import numpy as np
from datasets import load_dataset, DatasetDict, Dataset
from evaluate import load
from transformers import BertTokenizerFast, BertForSequenceClassification, TrainingArguments, Trainer

print(1)
path_data = r"D:\data\glue\rte"
path_model = r"D:\models\bert-base-cased"


# 加载训练数据、分词器、预训练模型以及评价方法
dataset:DatasetDict = load_dataset(path_data)
print(11)
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path=path_model)
print(12)
model     = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=path_model, num_labels=2, return_dict=True)
print(13)
# metric    = load(path=path_data)
# metric    = load('accuracy')

print(2)

# 对训练集进行分词
def tokenize(examples):
    return tokenizer(
        text=examples['sentence1'], 
        text_pair=examples['sentence2'], 
        truncation=True,     # 是否截断超长文本, 取值: bool(True表示截断所有超长值) / "only_first" / "only_second"
        padding='max_length' # 是否补齐到相同长度， bool（True 补到 batch 中最长） / "max_length"（补到 max_length） / "longest"
    )


dataset = dataset.map(tokenize, batched=True)   # 添加input_ids，token_type_ids, attention_mask字段
encoded_dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)  # 添加字段labels


print(3)
# 将数据集格式化为torch.Tensor类型以训练PyTorch模型
columns = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
encoded_dataset.set_format(type='torch', columns=columns)  # 将列columns按type类型输出


# # 定义评价指标
# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     return metric.compute(predictions=np.argmax(predictions, axis=1), references=labels)


print(4)
# 定义训练参数TrainingArguments，默认使用AdamW优化器
args = TrainingArguments(
    output_dir="ft-rte",                           # 输出路径，存放检查点和其他输出文件
    eval_strategy="epoch",        # 定义每轮结束后进行评价
    learning_rate=2e-5,                 # 定义初始学习率
    per_device_train_batch_size=16,     # 定义训练批次大小
    per_device_eval_batch_size=16,      # 定义测试批次大小
    num_train_epochs=2,                 # 定义训练轮数
    run_name='run_name_sjq'
)

print(5)
# 定义Trainer，指定模型和训练参数，输入训练集、验证集、分词器以及评价函数
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    processing_class=tokenizer,
    # compute_metrics=compute_metrics
)

print(6)
# 开始训练！（主流GPU上耗时约几小时）
trainer.train()



#   Attempting uninstall: sympy
#     Found existing installation: sympy 1.14.0
#     Uninstalling sympy-1.14.0:
#       Successfully uninstalled sympy-1.14.0
#   Attempting uninstall: torch
#     Found existing installation: torch 2.7.1+cu128
#     Uninstalling torch-2.7.1+cu128:
#       Successfully uninstalled torch-2.7.1+cu128
# WARNING: Ignoring invalid distribution -orch (c:\users\pc\miniconda3\envs\nlp\lib\site-packages)
# ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
# torchaudio 2.7.1+cu128 requires torch==2.7.1+cu128, but you have torch 2.6.0 which is incompatible.
# torchvision 0.22.1+cu128 requires torch==2.7.1+cu128, but you have torch 2.6.0 which is incompatible.
# Successfully installed accelerate-1.10.0 sympy-1.13.1 torch-2.6.0
# (nlp) PS D:\projects\transformer> pip install torch==2.7.1+cu128