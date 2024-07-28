#!/usr/bin/env bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER
export PYTHONPATH=$SHELL_FOLDER
echo "current path " $SHELL_FOLDER

# export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'

# declare an array
arr=("GLCTLVAML" "IVTDFSVIK" "TTDPSFLGRY" "AVFDRKSDAK" "GILGFVFTL" "KLGGALQAK" "LTDEMIAQY" "YLQPRTFLL" "RAKFKQLL" "ELAGIGILTV")
seeds=(27 28 29)
# for loop that iterates over each element in arr
for seed in "${seeds[@]}"
do
    # for loop that iterates over each element in arr
    for i in "${arr[@]}"
    do
        python3 /aceph/louisyuzhao/buddy2/linyangxiao/SC-AIR/SC-AIR-BERT-Multi/code/classification/train_multimodal.py \
    --vocab_path /aceph/louisyuzhao/buddy2/linyangxiao/SC-AIR/SC-AIR-BERT-Multi/data/vocab/vocab_2mer.pkl \
    --gene_token /aceph/louisyuzhao/buddy2/linyangxiao/SC-AIR/SC-AIR-BERT-Multi/data/classification/gene_full.csv \
    -c /aceph/louisyuzhao/buddy2/linyangxiao/SC-AIR/SC-AIR-BERT-Multi/extra/data/2-mer/$i/train_modified.tsv \
    -d /aceph/louisyuzhao/buddy2/linyangxiao/SC-AIR/SC-AIR-BERT-Multi/extra/data/2-mer/$i/valid_modified.tsv \
    -t /aceph/louisyuzhao/buddy2/linyangxiao/SC-AIR/SC-AIR-BERT-Multi/extra/data/2-mer/$i/test_modified.tsv \
    --bert_model /aceph/louisyuzhao/buddy2/linyangxiao/SC-AIR/back/xiaonasu_back/ImmuneBLAST_finalVersion/result/pretrain/ab_2mer_len81/hidden512_layers6_attn_heads4_batchsize1024_prob0.1_lr0.0001/bert.model.ep26 \
    -o /aceph/louisyuzhao/buddy2/linyangxiao/SC-AIR/SC-AIR-BERT-Multi/result/fintune/2-mer/$i/$seed \
    --lr_b 0.0001 \
    --lr_c 0.001 \
    --seq_len  79 \
    --prob 0.0 \
    --finetune 1 \
    --NNI_Search 0 \
    --class_name $i \
    --chain 2 \
    --batch_size 64 \
    --seed $seed
    done
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