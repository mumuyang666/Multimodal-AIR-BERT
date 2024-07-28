#!/usr/bin/env bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER
export PYTHONPATH=$SHELL_FOLDER
echo "current path " $SHELL_FOLDER

# 创建词库
python3.6 ../../code/bert/dataset/build_vocab.py \
-o test.pkl \
-k 1

# -o 输出地址
# -k kmer[1,2,3]