import csv
import os

input_folder = "/aceph/louisyuzhao/buddy2/linyangxiao/SC-AIR/SC-AIR-BERT-Multi/data/regression/DeepAIR_Affinity"  # 将此更改为包含输入TSV文件的文件夹路径

# 将字符串拆分为3个字符的子串
def split_into_3mers(s):
    return " ".join([s[i:i+3] for i in range(0, len(s) - 2)])

# 为每个输入文件生成一个转换后的TSV文件
for file_name in os.listdir(input_folder):
    if file_name.endswith(".tsv"):
        input_file = os.path.join(input_folder, file_name)
        output_file = f"m_{file_name}"
        
        # 读取输入TSV文件并转换格式
        converted_data = []
        with open(input_file, "r") as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter="\t")
            
            # 查找包含 "Reg" 的列名
            target_column = next(col for col in reader.fieldnames if "Reg" in col)
            
            for row in reader:
                converted_row = {
                    "ID": row["ID"],
                    "TRA_cdr3_3mer": split_into_3mers(row["TRA_cdr3"]),
                    "TRB_cdr3_3mer": split_into_3mers(row["TRB_cdr3"]),
                    "reg": row[target_column],
                    "TRA_cdr3": row["TRA_cdr3"],
                    "TRA_v_gene": row["TRA_v_gene"],
                    "TRA_j_gene": row["TRA_j_gene"],
                    "TRB_cdr3": row["TRB_cdr3"],
                    "TRB_v_gene": row["TRB_v_gene"],
                    "TRB_j_gene": row["TRB_j_gene"],
                }
                converted_data.append(converted_row)

        # 将转换后的数据写入新的TSV文件
        with open(output_file, "w") as tsvfile:
            fieldnames = ["ID", "TRA_cdr3_3mer", "TRB_cdr3_3mer", "reg", "TRA_cdr3", "TRA_v_gene", "TRA_j_gene", "TRB_cdr3", "TRB_v_gene", "TRB_j_gene"]
            writer = csv.DictWriter(tsvfile, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for row_data in converted_data:
                writer.writerow(row_data)

        print(f"已将转换后的数据保存到 {output_file}")