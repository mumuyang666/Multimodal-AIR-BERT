SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER
export PYTHONPATH=$SHELL_FOLDER
echo "current path " $SHELL_FOLDER

# export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'

# declare an array
arr=("GLCTLVAML" "IVTDFSVIK" "TTDPSFLGRY" "AVFDRKSDAK" "GILGFVFTL" "KLGGALQAK" "LTDEMIAQY" "YLQPRTFLL" "RAKFKQLL" "ELAGIGILTV")
# for loop that iterates over each element in arr

    # for loop that iterates over each element in arr
for i in "${arr[@]}"
    do
        python3 ./code/classification/train_multimodal.py \
    --vocab_path ./data/vocab/vocab_2mer.pkl \
    --gene_token ./data/classification/gene_full.csv \
    -c ./data/classification/$i/train_modified.tsv \
    -d ./data/classification/$i/valid_modified.tsv \
    -t ./data/classification/$i/test_modified.tsv \
    --bert_model ./checkpoint/pretrain_models/ab_3mer_len79.ep28 \
    -o ./result/fintune/$i/$seed \
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