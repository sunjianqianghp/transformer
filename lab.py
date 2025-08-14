import pandas as pd
import os 

cwd = os.path.dirname(__file__)

df = pd.read_parquet(os.path.join(cwd, 'data', 'glue', 'rte', 'test-00000-of-00001.parquet'))

print(df.head())


# Defined in Section 7.4.3.2

from os.path import join, dirname
import numpy as np

from datasets import load_dataset, DatasetDict, Dataset
from evaluate import load
print('note-1')
# from evaluate import load
print('note0')
from transformers import BertTokenizerFast, BertForSequenceClassification, TrainingArguments, Trainer

print('note1')
cwd = dirname(__file__)
path_data = join(cwd, '..', 'data', 'glue', 'rte')

print('note2')
# 加载训练数据
dataset:DatasetDict = load_dataset(path_data ) # Load a dataset from the Hugging Face Hub, or a local dataset.
print('note3')
print(dataset)

path_model = r'C:\Users\surface\projects\transformer\models\bert-base-cased'
# 加载分词器
tokenizer:BertTokenizerFast = BertTokenizerFast.from_pretrained(path_model)

# 加载预训练模型
model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=path_model, return_dict=True)
# # 加载评价方法
# metric = load('glue', 'rte')

# 对训练集进行分词
def tokenize(examples):
    return tokenizer(
        text=examples['sentence1'],  
        text_pair=examples['sentence2'], 
        truncation=True,     # 是否截断超长文本, 取值: bool(True表示截断所有超长值) / "only_first" / "only_second"
        padding='max_length' # 是否补齐到相同长度， bool（True 补到 batch 中最长） / "max_length"（补到 max_length） / "longest"
    )

dataset = dataset.map(tokenize, batched=True) # 添加input_ids，token_type_ids, attention_mask字段

dataset_validation:Dataset = dataset['validation']
print( dataset_validation.data.to_pandas()[['label', 'idx', 'input_ids']] )


print(type(dataset))
print(dataset['validation'])


encoded_dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True) # 添加字段labels

encoded_dataset_validation:Dataset = encoded_dataset['validation']
print( encoded_dataset_validation.data.to_pandas()[['label', 'idx', 'input_ids', 'labels']] )

print( encoded_dataset_validation.format )



# 将数据集格式化为torch.Tensor类型以训练PyTorch模型
columns = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
encoded_dataset.set_format(type='torch', columns=columns) # 将列columns按type类型输出

metric = load(path="glue", config_name="rte")
# 定义评价指标
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return metric.compute(predictions=np.argmax(predictions, axis=1), references=labels)

# 定义训练参数TrainingArguments，默认使用AdamW优化器
args = TrainingArguments(
    "ft-rte",                           # 输出路径，存放检查点和其他输出文件
    evaluation_strategy="epoch",        # 定义每轮结束后进行评价
    learning_rate=2e-5,                 # 定义初始学习率
    per_device_train_batch_size=16,     # 定义训练批次大小
    per_device_eval_batch_size=16,      # 定义测试批次大小
    num_train_epochs=2,                 # 定义训练轮数
)

# 定义Trainer，指定模型和训练参数，输入训练集、验证集、分词器以及评价函数
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 开始训练！（主流GPU上耗时约几小时）
trainer.train()



