#!/usr/bin/env bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER
export PYTHONPATH=$SHELL_FOLDER
echo "current path " $SHELL_FOLDER

# export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
# "GLCTLVAML" "IVTDFSVIK" "AVFDRKSDAK" "GILGFVFTL" "KLGGALQAK" "RAKFKQLL" "ELAGIGILTV" "eos"
# declare an array
arr=("GLCTLVAML" )

# for loop that iterates over each element in arr
for i in "${arr[@]}"
do  
    python3 /aceph/louisyuzhao/buddy2/linyangxiao/SC-AIR/SC-AIR-BERT-Multi/code/classification/train_dual.py \
--vocab_path /aceph/louisyuzhao/buddy2/linyangxiao/SC-AIR/SC-AIR-BERT-Multi/data/vocab/vocab_3mer.pkl \
--gene_token /aceph/louisyuzhao/buddy2/linyangxiao/SC-AIR/SC-AIR-BERT-Multi/data/classification/gene_full.csv \
-c /aceph/louisyuzhao/buddy2/linyangxiao/SC-AIR/SC-AIR-BERT-Multi/data/regression/DeepAIR_Affinity/modify_/m_${i}_train.tsv \
-d /aceph/louisyuzhao/buddy2/linyangxiao/SC-AIR/SC-AIR-BERT-Multi/data/regression/DeepAIR_Affinity/modify_/m_${i}_val.tsv \
-t /aceph/louisyuzhao/buddy2/linyangxiao/SC-AIR/SC-AIR-BERT-Multi/data/regression/DeepAIR_Affinity/modify_/m_${i}_test.tsv \
--bert_model /aceph/louisyuzhao/buddy2/linyangxiao/SC-AIR/SC-AIR-BERT-Multi/checkpoint/pretrain_models/ab_3mer_len79.ep28 \
-o /aceph/louisyuzhao/buddy2/linyangxiao/SC-AIR/SC-AIR-BERT-Multi/result_NNI/reg_dual/test/$i \
--lr_b 0.0001 \
--lr_c 0.001 \
--seq_len  79 \
--prob 0.0 \
--finetune 1 \
--NNI_Search 1 \
--class_name $i \
--chain 2 \
--batch_size 64 \
--seed 27

done

# --vocab_path 词库
# -c 训练数据集
# -d 验证数据集
# -t 测试数据集
# --bert_model 预训练模型
# --output_path 输出地址
# --lr_b 微调BERT的学习率
# --lr_c 全连接层的学习率
# --seq_len 数据总长（<sos>TRA_cdr3<eos>TRB_cdr3<eos><pad>）
# --prob mask probability
# --finetune 是否微调[0,1]
# --NNI_Search [0,1]
# --in_features bert的hidden
# --class_name 分类的正样本标签
# --chain [1,2] 单链还是双链