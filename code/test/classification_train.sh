#!/usr/bin/env bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER
export PYTHONPATH=$SHELL_FOLDER
echo "current path " $SHELL_FOLDER

export CUDA_VISIBLE_DEVICES='1'

python3 ../classification/train_multimodal.py \
--vocab_path ../../data/vocab/vocab_3mer.pkl \
-c ../../data/classification/Plasmodium_seed-15/train_modified.tsv \
-d ../../data/classification/Plasmodium_seed-15/valid_modified.tsv \
-t ../../data/classification/Plasmodium_seed-15/test_modified.tsv \
--bert_model ../../checkpoint/pretrain_models/ab_3mer_len79.ep28 \
-o ../../result/classification/test/Plasmodium \
--lr_b 0.0001 \
--lr_c 0.001 \
--seq_len  79 \
--prob 0.0 \
--finetune 1 \
--NNI_Search 1 \
--class_name 1 \
--chain 2 \
--batch_size 32 \
--seed 27


# 微调BERT && 训练分类器
# python3 ../classification/train_multimodal.py \
# --vocab_path ../../data/vocab/vocab_3mer.pkl \
# -c ../../data/classification/10x/10x_ab_3mer_7_2_1/RAKFKQLL/train_modified.tsv \
# -d ../../data/classification/10x/10x_ab_3mer_7_2_1/RAKFKQLL/valid_modified.tsv \
# -t ../../data/classification/10x/10x_ab_3mer_7_2_1/RAKFKQLL/test_modified.tsv \
# --bert_model ../../checkpoint/pretrain_models/ab_3mer_len79.ep28 \
# -o ../../result/classification/test \
# --lr_b 0.0001 \
# --lr_c 0.001 \
# --seq_len  79 \
# --prob 0.0 \
# --finetune 1 \
# --NNI_Search 0 \
# --class_name RAKFKQLL \
# --chain 2 \
# --batch_size 32 \
# --seed 27

# python3 ../classification/train.py \
# --vocab_path ../../data/vocab/vocab_3mer.pkl \
# -c ../../data/classification/Ebola_seed-15/train.tsv \
# -d ../../data/classification/Ebola_seed-15/valid.tsv \
# -t ../../data/classification/Ebola_seed-15/test.tsv \
# --bert_model ../../checkpoint/pretrain_models/ab_3mer_len79.ep28 \
# -o ../../result/classification/test \
# --lr_b 0.0001 \
# --lr_c 0.001 \
# --seq_len  79 \
# --prob 0.0 \
# --finetune 1 \
# --NNI_Search 0 \
# --in_features 512 \
# --class_name 1 \
# --chain 2 \
# --batch_size 32 \
# --seed 27


# --in_features 512 \

# --vocab_path 词库
# -c 训练数据集
# -d 验证数据集
# -t 测试数据集
# --bert_model 预训练模型
# -o 输出地址
# --lr_b 微调BERT的学习率
# --lr_c 全连接层的学习率
# --seq_len 数据总长（<sos>TRA_cdr3<eos>TRB_cdr3<eos><pad>）
# --prob mask probability（分类任务时为0）
# --finetune 是否微调[0,1]
# --NNI_Search [0,1]
# --in_features bert的hidden
# --class_name 分类的正样本标签
# --chain [1,2] 单链还是双链