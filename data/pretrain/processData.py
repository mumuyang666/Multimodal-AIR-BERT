import pandas as pd
import os
import os.path
import numpy as np

# 将/rawdata/pretrain数据处理成/data/pretrain数据
# ！！！！！！别轻易运行，改一下输出路径output_path，不然把原来的数据覆盖了
# 输入：TRA_cdr3,TRB_cdr3；输出：BERT预训练的输入数据

# 从TRA_cdr3,TRB_cdr3 处理，得到BERT的输入数据（normal,3mer)
def process(input_path,output_path1,output_path2,test_size=0.2):
    
    dirs = os.listdir(input_path)

    a_name = 'TRA_cdr3'
    b_name = 'TRB_cdr3'

    data = pd.DataFrame(columns=[a_name,b_name])
    max,max_3mer = 0,0
    
    TRA_cdr3,TRB_cdr3 = [],[]
    TRA_3mer_cdr3,TRB_3mer_cdr3 = [],[]
    sequence_characters = []

    count = 0
    # 分割cdr3的氨基酸
    for file in dirs:
        df = pd.read_csv(os.path.join(input_path,file))

        for i,row in df.iterrows():
            if((len(row[a_name])>2) and (len(row[b_name])>2)):
                a,b ='',''
                # normal
                for i in range(len(row[a_name])):
                    a += row[a_name][i]
                    if(i!=len(row[a_name])-1):
                        a += ' '
                
                for i in range(len(row[b_name])):
                    b += row[b_name][i]
                    if(i!=len(row[b_name])-1):
                        b += ' '
                
                TRA_cdr3.append(a)
                TRB_cdr3.append(b)
                if(len(row[a_name])+len(row[b_name])>max):
                    max = len(row[a_name]) + len(row[b_name])
                
                c,d ='',''
                # 3mer
                for i in range(len(row[a_name])-2):
                    a_mer = row[a_name][i] + row[a_name][i+1] + row[a_name][i+2]
                    c += a_mer
                    if(a_mer not in sequence_characters):
                        sequence_characters.append(a_mer)
                    if(i!=len(row[a_name])-3):
                        c += ' '
                
                for i in range(len(row[b_name])-2):
                    b_mer = row[b_name][i] + row[b_name][i+1] + row[b_name][i+2]
                    d += b_mer
                    if(b_mer not in sequence_characters):
                        sequence_characters.append(b_mer)
                    if(i!=len(row[b_name])-3):
                        d += ' '
        
                TRA_3mer_cdr3.append(c)
                TRB_3mer_cdr3.append(d)
                if(len(row[a_name])-2+len(row[b_name])-2>max_3mer):
                    max_3mer = len(row[a_name])-2+len(row[b_name])-2

    print(sequence_characters)
    print('sequence_characters length:',len(sequence_characters))
    print('max length:',max)
    print('max_3mer length:',max_3mer)
    data = {
        'TRA_cdr3':TRA_cdr3,
        'TRB_cdr3':TRB_cdr3,
        'TRA_3mer_cdr3':TRA_3mer_cdr3,
        'TRB_3mer_cdr3':TRB_3mer_cdr3
    }
    cdr3 = pd.DataFrame(data)
    print(cdr3)
    cdr3[['TRA_cdr3','TRB_cdr3']].to_csv(os.path.join(output_path1,'full_data.tsv'),sep='\t',index=None,header=None)
    cdr3[['TRA_3mer_cdr3','TRB_3mer_cdr3']].to_csv(os.path.join(output_path2,'full_data.tsv'),sep='\t',index=None,header=None)

    # split train dataset; test dataset
    idx = np.array(range(len(cdr3)))
    test_size = test_size
    np.random.shuffle(idx)
    train_idx = np.random.choice(idx, int((1 - test_size) * idx.shape[0]), replace=False)
    test_idx = np.setdiff1d(idx, train_idx)

    train_dataset = cdr3.iloc[train_idx]
    test_dataset = cdr3.iloc[test_idx]

    train_dataset[['TRA_cdr3','TRB_cdr3']].to_csv(os.path.join(output_path1,'train.tsv'),sep='\t',index=None,header=None)
    test_dataset[['TRA_cdr3','TRB_cdr3']].to_csv(os.path.join(output_path1,'test.tsv'),sep='\t',index=None,header=None)

    train_dataset[['TRA_3mer_cdr3','TRB_3mer_cdr3']].to_csv(os.path.join(output_path2,'train.tsv'),sep='\t',index=None,header=None)
    test_dataset[['TRA_3mer_cdr3','TRB_3mer_cdr3']].to_csv(os.path.join(output_path2,'test.tsv'),sep='\t',index=None,header=None)

# 从TRB_cdr3 处理，得到BERT的输入数据（train,test)
def single_process(input_path,output_path1,output_path2,chain,test_size=0.2):
    # all data
    
    dirs = os.listdir(input_path)
    a_name = chain
    data = pd.DataFrame(columns=[a_name])
    max,max_3mer = 0,0
    TRB_cdr3,TRB_3mer_cdr3 = [],[]
    sequence_characters = []
    temp = ''

    # total_epitope = []
    count = 0
    # 分割cdr3的氨基酸
    for file in dirs:
        df = pd.read_csv(os.path.join(input_path,file))

        for i,row in df.iterrows():
            if(len(row[a_name])>2):
                a =''
                for i in range(len(row[a_name])):
                    a += row[a_name][i]
                    if(i!=len(row[a_name])-1):
                        a += ' '
                TRB_cdr3.append(a)
                if(len(row[a_name])>max):
                    max = len(row[a_name])
                    temp = row[a_name]

                b =''
                for i in range(len(row[a_name])-2):
                    b_mer = row[a_name][i] + row[a_name][i+1] + row[a_name][i+2]
                    b += b_mer
                    if(b_mer not in sequence_characters):
                        sequence_characters.append(b_mer)
                    if(i!=len(row[a_name])-3):
                        b += ' '

            TRB_3mer_cdr3.append(b)
            if(len(row[a_name])-2>max_3mer):
                max_3mer = len(row[a_name])-2

        data = {
            'TRB_cdr3':TRB_cdr3,
            'TRB_3mer_cdr3':TRB_3mer_cdr3
        }
        cdr3 = pd.DataFrame(data)

    print("max_length:",max)
    print("max_3mer_length:",max_3mer)
    cdr3[['TRB_cdr3']].to_csv(os.path.join(output_path1,'full_data.tsv'),sep='\t',index=None,header=None)
    cdr3[['TRB_3mer_cdr3']].to_csv(os.path.join(output_path2,'full_data.tsv'),sep='\t',index=None,header=None)

    # cdr3 =pd.read_csv('/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/data/covid_19/full_data.tsv',sep='\t')

    # split train dataset; test dataset
    idx = np.array(range(len(cdr3)))
    test_size = test_size
    np.random.shuffle(idx)
    train_idx = np.random.choice(idx, int((1 - test_size) * idx.shape[0]), replace=False)
    test_idx = np.setdiff1d(idx, train_idx)

    train_dataset = cdr3.iloc[train_idx]
    test_dataset = cdr3.iloc[test_idx]

    train_dataset[['TRB_cdr3']].to_csv(os.path.join(output_path1,'train.tsv'),sep='\t',index=None,header=None)
    test_dataset[['TRB_cdr3']].to_csv(os.path.join(output_path1,'test.tsv'),sep='\t',index=None,header=None)

    train_dataset[['TRB_3mer_cdr3']].to_csv(os.path.join(output_path2,'train.tsv'),sep='\t',index=None,header=None)
    test_dataset[['TRB_3mer_cdr3']].to_csv(os.path.join(output_path2,'test.tsv'),sep='\t',index=None,header=None)

def onmer_to_twoMer():
    input_path = '/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/data/10x_NPC_IBD_IEDB_VDJ/normal_8_2'
    output_path = '/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/data/10x_NPC_IBD_IEDB_VDJ/2mer_8_2'
    dirs = os.listdir(input_path)
    for file in dirs:
        input_file = os.path.join(input_path,file)
        df = pd.read_csv(input_file,sep='\t',header=None)
        two_mer_a = []
        two_mer_b = []
        for i,row in df.iterrows():
            one_mer_a = row[0].split()
            one_mer_b = row[1].split()
            # 处理a链
            two_mer = ''
            for j in range(len(one_mer_a)-1):
                two_mer += one_mer_a[j]+one_mer_a[j+1]
                if(j!=len(one_mer_a)-2):
                    two_mer += ' '
            two_mer_a.append(two_mer)
            # 处理b链
            two_mer = ''
            for j in range(len(one_mer_b)-1):
                two_mer += one_mer_b[j]+one_mer_b[j+1]
                if(j!=len(one_mer_b)-2):
                    two_mer += ' '
            two_mer_b.append(two_mer)
        data ={
            'a':two_mer_a,
            'b':two_mer_b
        }
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(output_path,file),header=None,index=None,sep='\t')
        
if __name__=='__main__':


    # 处理bert_ab的输入数据
    # input_path = '/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/data/ab_unique/10x_NPC_IBD_VDJ_IEDB_ab'
    # output_path1 = '/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/data/10x_NPC_IBD_IEDB_VDJ/normal_8_2'
    # output_path2 = '/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/data/10x_NPC_IBD_IEDB_VDJ/3mer_8_2'
    # process(input_path,output_path1,output_path2)

    # 处理bert_b的输入数据
    # input_path = '/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/data/ab_unique/10x_NPC_IBD_VDJ_IEDB_b'
    # output_path1 = '/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/data/10x_NPC_IBD_IEDB_VDJ_b/normal_8_2'
    # output_path2 = '/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/data/10x_NPC_IBD_IEDB_VDJ_b/3mer_8_2'
    # single_process(input_path,output_path1,output_path2,chain='TRA_cdr3')

    # 处理bert_be的输入数据
    # input_path = '/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/data/ab_unique/10x_VDJ_IEDB_be'
    # output_path1 = '/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/data/10x_VDJ_IEDB_be/normal_8_2'
    # output_path2 = '/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/data/10x_VDJ_IEDB_be/3mer_8_2'
    # process(input_path,output_path1,output_path2)

    # 处理bert_a的输入数据
    # input_path = '/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/data/ab_unique/10x_NPC_IBD_VDJ_IEDB_a'
    # output_path1 = '/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/data/10x_NPC_IBD_IEDB_VDJ_a/normal_8_2'
    # output_path2 = '/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/data/10x_NPC_IBD_IEDB_VDJ_a/3mer_8_2'
    # single_process(input_path,output_path1,output_path2,chain='TRA_cdr3')

    



    
