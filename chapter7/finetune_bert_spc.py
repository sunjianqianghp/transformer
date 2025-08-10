# Defined in Section 7.4.3.2

# import numpy as np
# from datasets import load_dataset, load_metric
# from transformers import BertTokenizerFast, BertForSequenceClassification, TrainingArguments, Trainer


# # 加载训练数据
# dataset = load_dataset(path='glue', name='rte') # Load a dataset from the Hugging Face Hub, or a local dataset.
# # 加载分词器
# tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
# # 加载预训练模型
# model = BertForSequenceClassification.from_pretrained('bert-base-cased', return_dict=True)
# # 加载评价方法
# metric = load_metric('glue', 'rte')



from huggingface_hub import list_datasets

# List first 10 datasets
datasets = list_datasets(limit=10)
for ds in datasets:
    print(ds.id)



# https://keke.kkhhddnn.cn/api/v1/client/subscribe?token=e3151a1a63bcb9c840c0dc66dc3f1a22