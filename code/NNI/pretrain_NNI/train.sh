#!/usr/bin/env bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER
export PYTHONPATH=$SHELL_FOLDER
echo "current path " $SHELL_FOLDER

export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'

# NNI
python3 ../../bert/main_MLM_NNI.py \
-c ../../../data/pretrain/10x-NPC-IBD-IEDB-VDJDB_ab/3mer_8_2/train.tsv \
-t ../../../data/pretrain/10x-NPC-IBD-IEDB-VDJDB_ab/3mer_8_2/test.tsv \
--vocab_path ../../../data/vocab/vocab_3mer.pkl \
--output_path ../../../result/pretrain/test \
--seq_len  79 \
--num_workers 32 \
--embedding_mode normal

