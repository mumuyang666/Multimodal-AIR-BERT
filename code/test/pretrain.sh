#!/usr/bin/env bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER
export PYTHONPATH=$SHELL_FOLDER
echo "current path " $SHELL_FOLDER

export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'

# 预训练
python3 ../bert/main_MLM_NNI.py \
-c ../../data/pretrain/10x-NPC-IBD-IEDB-VDJDB_ab/3mer_8_2/train.tsv \
-t ../../data/pretrain/10x-NPC-IBD-IEDB-VDJDB_ab/3mer_8_2/test.tsv \
--vocab_path ../../data/vocab/vocab_3mer.pkl \
--output_path ../../result/pretrain/test \
--seq_len  79 \
--num_workers 32 \
--embedding_mode normal \
--NNI_Search 0 \
--lr 0.0001 \
--epochs 50 \
--hidden 512 \
--layers 6 \
--attn_heads 4 \
--batch_size 1024 \
--prob 0.1 \
--process_mode MLM

# -c 预训练-训练数据集
# -t 预训练-验证数据集
# --vocab_path 词库
# --output_path 输出地址
# --seq_len 数据总长（<sos>TRA_cdr3<eos>TRB_cdr3<eos><pad>）
# --embedding_mode 可选值 ['normal','atchley','kidera','onehot']
# --NNI_Search [0,1]
# --lr learnning rate
# --prob mask probability

# --process_mode 可选值['MLM','MLM_MN','MLM_SS','MLM_SS_MN'] 
# 'MLM'：双链正常mask
# 'MLM_MN':双链mask maskNeighbours
# 'MLM_SS':单链正常mask
# 'MLM_SS_MN':单链mask maskNeighbours