#!/usr/bin/env bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER
export PYTHONPATH=$SHELL_FOLDER
echo "current path " $SHELL_FOLDER

export CUDA_VISIBLE_DEVICES='0'

python3 ../classification/train_titan.py \
--vocab_path ../../data/vocab/vocab_3mer.pkl \
--bert_model ../../result/pretrain/models/be_3mer_len84.ep33 \
-o ../../result/classification/test_titan \
--seq_len  84 \
--lr_c 0.001 \
--lr_b 0.0001 \
--class_name 1 \
--prob 0.0 \
--finetune 1 \
--in_features 512 \
--batch_size 512 \
--input_path  ../../data/classification/TITAN/vdj_covid/vdj_covid_strict_split_3mer/fold0