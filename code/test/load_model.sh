#!/usr/bin/env bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER
export PYTHONPATH=$SHELL_FOLDER
echo "current path " $SHELL_FOLDER

export CUDA_VISIBLE_DEVICES='0'

# 加载使用训练好的分类器
python3 ../classification/load_model.py \
--vocab_path ../../data/vocab/vocab_3mer.pkl \
--test_dataset ../../data/classification/10x/10x_ab_3mer_7_2_1/KLGGALQAK/test.tsv \
--seq_len 79 \
--class_name KLGGALQAK \
--load_model ../../result/classification/ab_3mer_len79/10x/KLGGALQAK_semi/max_auc_model.pth \
-o ../../result/classification/test/test.csv

# --vocab_path 词库
# --test_dataset 测试数据集
# --seq_len 数据总长（<sos>TRA_cdr3<eos>TRB_cdr3<eos><pad>）
# --class_name 分类的正样本标签
# --load_model 分类器模型

# 计算输出attention
# python3 ../classification/load_model.py \
# --vocab_path ../../data/vocab/vocab_3mer.pkl \
# --test_dataset /aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/result/heatmap/RAKFKQLL_11_13.tsv \
# --seq_len 79 \
# --class_name RAKFKQLL \
# --load_model /aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST_finalVersion/result/classification/ab_3mer_len79/10x/RAKFKQLL/max_auc_model.pth