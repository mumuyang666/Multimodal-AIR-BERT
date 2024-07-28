import os
import pandas as pd
import glob

def merge_performance(input_files, output_file):
    output_df = pd.DataFrame()
    for curr_file in input_files:
        curr_df = pd.read_csv(curr_file)
        # Assuming the folder name is the second last element in the file path when split by '/'
        curr_df['source_folder'] = curr_file.split('/')[-2]
        output_df = pd.concat([output_df, curr_df], axis=0)

    output_df.to_csv(output_file)
    
    
if __name__ == "__main__":
    key_list = ["GLCTLVAML","IVTDFSVIK" ,"TTDPSFLGRY", "AVFDRKSDAK", "GILGFVFTL", "KLGGALQAK" ,"LTDEMIAQY", "YLQPRTFLL", "RAKFKQLL", "ELAGIGILTV" ]
    for key in key_list:
        input_folder = f'/aceph/louisyuzhao/buddy2/linyangxiao/SC-AIR/SC-AIR-BERT-Multi/result_NNI/classification_LMF/{key}/{key}'
        output_file = f'/aceph/louisyuzhao/buddy2/linyangxiao/SC-AIR/SC-AIR-BERT-Multi/result_NNI/classification_LMF/{key}/{key}_performance.csv'
        input_files = glob.glob(os.path.join(input_folder, '*/parameters.csv'))
        merge_performance(input_files, output_file)