from typing import List
import random


class MaskedLmInstance:
    def __init__(self, index, label):
        self.index = index 
        self.label = label




def create_masked_lm_predictions(
        tokens:List, 
        masked_lm_prob, 
        max_predictions_per_seq, 
        vocab_words
    ):
    """
    此函数用户创建MLM任务的训练数据
    tokens: 输入文本
    masked_lm_prob: 掩码语言模型的掩码概率
    max_predictions_per_seq: 每个序列的最大预测数目
    vocab_words: 词表列表
    rng: 随机数生成器
    """
    cand_indexes = [] # 存储可以参与掩码的token(不含"[CLS]"和"[SEP]")下标, 互不相同, 被打乱
    for (i, token) in enumerate(tokens):
        # 掩码时跳过[CLS]和[SEP]
        if token == "[CLS]" or token == "[SEP]":
            continue
        cand_indexes.append(i)
    random.shuffle(cand_indexes) # 随机打乱所有下标

    num_to_predict = min(
        max_predictions_per_seq, 
        max(1, int(round(len(tokens)*masked_lm_prob)) )
    ) # 计算预测数目（掩码数量k）
    
    output_tokens = list(tokens) # 存储掩码后的输入序列x^'_1 x^'_2 ... x^'_n，初始化为原始输入
    masked_lms = [] # 存储掩码实例
    covered_indexes = set() # 存储已经处理过的下标
    for cand_index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if cand_index in covered_indexes:
            continue
        covered_indexes.add(cand_index)

        # 以80%的概率替换为 "[MASK]"
        if random.random()<0.8:
            masked_token = "[MASK]"
        else:
            # 以10%的概率不进行任何替换
            if random.random() < 0.5:
                masked_token = tokens[cand_index] 
            # 以10%的概率替换成词表中的"随机词"
            else:
                masked_token = vocab_words[random.randint( 0, len(vocab_words)-1 )]
        
        output_tokens[cand_index] = masked_token # 设置为被掩码的token
        masked_lms.append(MaskedLmInstance( index=cand_index, label=tokens[cand_index] ))

    masked_lms = sorted(masked_lms, key=lambda x:x.index) # 按下标升序排列

    masked_lm_positions = [] # 存储需要掩码的下标
    masked_lm_labels    = [] # 存储掩码前的原词，即还原目标
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    
    return output_tokens, masked_lm_positions, masked_lm_labels

    




    
    
