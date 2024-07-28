import os
import glob
import pandas as pd


def generate_combined_table(input_file_list, reference_file):
    
    reference_df = pd.read_csv(reference_file)
    ref_df_selected = reference_df[['ID','TRA_cdr3','TRA_v_gene','TRA_j_gene','TRB_cdr3','TRB_v_gene','TRB_j_gene']]
    for input_file in input_file_list:
        input_df = pd.read_csv(input_file, sep='\t')
        input_df.columns =['ID','TRA_cdr3_3Mer','TRB_cdr3_3Mer','epitope']        
        input_df_merge = input_df.merge(ref_df_selected, on='ID', how='inner')
        
        Dname = os.path.dirname(input_file)
        Bname = os.path.basename(input_file).split('.')[0]
        output_file = os.path.join(Dname,Bname+"_modified.tsv")
        input_df_merge.to_csv(output_file, sep='\t', index=None, header=None)
        
if __name__ == '__main__':
    input_file = '/aaa/louisyuzhao/project1/SC-AIR-BERT/data/classification/Influenza_A/Influenza_A_seed-7_3mer_7_2_1'
    reference_file = '/aaa/louisyuzhao/project1/SC-AIR-BERT/rawdata/classification/bcell_receptor_combine_removeNoAntigen_removeNoStructure_drop_duplicated.csv'
    extItem = '*.tsv'
    input_file_list = glob.glob(os.path.join(input_file, extItem))
    
    generate_combined_table(input_file_list, reference_file)