import csv
import os

input_folder = "/aceph/louisyuzhao/buddy2/linyangxiao/SC-AIR/SC-AIR-BERT-Multi/data/regression/DeepAIR_Affinity"  # 将此更改为包含输入文件的文件夹路径

# 从CSV文件中提取所需的列
def extract_columns(file_name, target_column):
    data = []
    with open(file_name, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append({
                "TRB_v_gene": row["TRB_v_gene"],
                "TRB_j_gene": row["TRB_j_gene"],
                "TRA_v_gene": row["TRA_v_gene"],
                "TRA_j_gene": row["TRA_j_gene"],
                "TRB_cdr3": row["TRB_cdr3"],
                "TRA_cdr3": row["TRA_cdr3"],
                "ID": row["ID"],
                target_column: row[target_column],
            })
    return data

# 为每个输入文件生成一个TSV文件
for file_name in os.listdir(input_folder):
    if file_name.endswith(".csv"):
        file_path = os.path.join(input_folder, file_name)
        
        # 提取目标列名
        target_column = file_name.split("_Reg")[0] + "_Reg"
        
        data = extract_columns(file_path, target_column)
        
        # 构造输出文件名
        file_name_parts = file_name.split("_")
        antigen = file_name_parts[1]
        dataset_type = file_name_parts[-2]
        output_file = f"{antigen}_{dataset_type}.tsv"
        
        # 将提取的数据写入新的TSV文件
        with open(output_file, "w") as tsvfile:
            fieldnames = ["TRB_v_gene", "TRB_j_gene", "TRA_v_gene", "TRA_j_gene", "TRB_cdr3", "TRA_cdr3", "ID", target_column]
            writer = csv.DictWriter(tsvfile, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for row_data in data:
                writer.writerow(row_data)

        print(f"已将提取的数据保存到 {output_file}")