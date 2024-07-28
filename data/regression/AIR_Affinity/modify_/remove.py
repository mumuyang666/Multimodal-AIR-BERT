import os

def remove_header_from_tsv(dir_path):
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".tsv"):
                file_path = os.path.join(root, file)
                
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                with open(file_path, 'w') as f:
                    # 跳过第一行（表头）并写回文件
                    f.writelines(lines[1:])

# 使用方法
dir_path = "/aceph/louisyuzhao/buddy2/linyangxiao/SC-AIR/SC-AIR-BERT-Multi/data/regression/DeepAIR_Affinity/modify_"  # 请替换为您的路径
remove_header_from_tsv(dir_path)