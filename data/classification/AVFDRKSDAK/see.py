import pandas as pd

data = pd.read_csv('/aceph/louisyuzhao/buddy2/linyangxiao/SC-AIR/SC-AIR-BERT-Multi/data/classification/AVFDRKSDAK/test.tsv', sep='\t', encoding='utf-8')

# 查看数据的前几行
print(data.head())