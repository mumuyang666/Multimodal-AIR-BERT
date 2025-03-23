SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER
export PYTHONPATH=$SHELL_FOLDER
echo "current path " $SHELL_FOLDER

export CUDA_VISIBLE_DEVICES='0'

# 加载使用训练好的分类器
python3 ./code/classification/inference.py \
--vocab_path ./data/vocab/vocab_3mer.pkl \
--gene_token ./data/classification/gene_full.csv \
--bert_model ./checkpoint/pretrain_models/ab_3mer_len79 \
--test_dataset ./data/classification/KLGGALQAK/test_modified.tsv \
--seq_len 79 \
--class_name KLGGALQAK \
--load_model ./checkpoint/finetune_models/KLGGALQAK/model.pth \
-o ../../result/classification/test/test.csv