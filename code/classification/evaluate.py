import os
from os import system
import os.path
from os.path import isdir
import sys
sys.path.append(os.path.abspath('/aceph/louisyuzhao/buddy2/linyangxiao/SC-AIR/SC-AIR-BERT-Multi/code'))

import torch
from torch.utils.data import DataLoader
from Dataset_reg import DatasetMultimodal
from Dataset_singleSentence import Dataset_singleSentence

from bert.dataset import WordVocab


from gene.gene_encoder import get_gene_token, save_gene_token, load_saved_token
from gene.gene_encoder import get_gene_encoder

import argparse
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd


import logging
import random

import matplotlib.pyplot as plt

# 设置参数
corpus_path = "/aceph/louisyuzhao/buddy2/linyangxiao/SC-AIR/SC-AIR-BERT-Multi/data/regression/DeepAIR_Affinity/modify_/m_AVFDRKSDAK_test.tsv"  # 更改为您的测试数据文件路径
vocab_path = "/aceph/louisyuzhao/buddy2/linyangxiao/SC-AIR/SC-AIR-BERT-Multi/data/vocab/vocab_3mer.pkl"       # 更改为您的词汇表文件路径
seq_len = 79                           # 根据您的需求设置序列长度

# 加载词汇表
vocab = WordVocab.load_vocab(vocab_path)

# 创建 DatasetMultimodal 实例
dataset = DatasetMultimodal(corpus_path, vocab, seq_len, class_name="your_class_name")

# 检查数据集中的前 n 个样本
# 检查整个数据集
total_lines = len(dataset.lines) if dataset.on_memory else "Unknown (Not loaded into memory)"

print(f"Total lines in dataset: {total_lines}")

problematic_lines = []

for i in range(len(dataset.lines)):  # 如果数据不在内存中，您可能需要更改这一部分
    try:
        data = dataset[i]
        # 可以选择是否打印每一行的数据，为了简洁性，我在这里注释掉了
        # print(f"Sample {i}:")
        # print("ID:", data["ID"])
        # ... 其他字段 ...

    except Exception as e:
        print(f"Error on line {i}: {e}")
        problematic_lines.append(i)

print("\nProblematic Lines:")
for line_num in problematic_lines:
    print(f"Line {line_num}: {dataset.lines[line_num]}")

with open(corpus_path, 'r', encoding='utf-8') as f:
    print("Actual lines in file:", sum(1 for _ in f))