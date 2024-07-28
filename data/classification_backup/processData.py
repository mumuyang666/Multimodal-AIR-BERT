import pandas as pd
import os
import os.path
import numpy as np
from os.path import isdir
from os import mkdir,system
from sklearn import metrics

# 将/rawdata/classification数据处理成/data/classification数据
# ！！！！！！别轻易运行，改一下输出路径output_path，不然把原来的数据覆盖了

# unique TRA_cdr3,TRB_cdr3去重
def Unique(input_path,output_path):

    path = input_path
    dirs = os.listdir(path)
    TRA_cdr3 = []
    TRB_cdr3= []
    epitopes = []
    disease = []
    for file in dirs:
        epitope = file.split("_")[1].split('.')[0]
        df = pd.read_csv(os.path.join(path,file))
        TRA_cdr3 += df['TRA_cdr3'].tolist()
        TRB_cdr3 += df['TRB_cdr3'].tolist()
        disease += df['class'].tolist()
        epitopes += [epitope]*len(df)
    data = {
        'TRA_cdr3':TRA_cdr3,
        'TRB_cdr3':TRB_cdr3,
        'disease':disease
    }
    cdr3 = pd.DataFrame(data)
    cdr3.drop_duplicates(inplace=True)
    # cdr3.drop_duplicates(['TRA_cdr3','TRB_cdr3'],inplace=True)
    print(cdr3)
    cdr3.to_csv(output_path,index=None)

def Split(input_file,output_path):

    df = pd.read_csv(input_file)
    a_name = 'TRA_cdr3'
    b_name = 'TRB_cdr3'
    data = pd.DataFrame(columns=[a_name,b_name])
    max = 0
    TRA_cdr3,TRB_cdr3 = [],[]
    for i,row in df.iterrows():
        a,b ='',''
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

    data = {
        'TRA_cdr3':TRA_cdr3,
        'TRB_cdr3':TRB_cdr3
    }
    cdr3 = pd.DataFrame(data)
    cdr3['label'] = df['disease'].tolist()
    print("max_length:",max)
    print(cdr3)
    
    output_path = output_path + '_maxlen' + str(max)
    if not(isdir(output_path)):
        mkdir(output_path)
    cdr3.to_csv(os.path.join(output_path,'full_data.tsv'),sep='\t',index=None,header=None)

    # split train dataset; test dataset
    idx = np.array(range(len(cdr3)))
    test_size = 0.3
    np.random.shuffle(idx)
    train_idx = np.random.choice(idx, int((1 - test_size) * idx.shape[0]), replace=False)
    test_idx = np.setdiff1d(idx, train_idx)

    train_dataset = cdr3.iloc[train_idx]
    test_dataset = cdr3.iloc[test_idx]

    train_dataset.to_csv(os.path.join(output_path,'train.tsv'),sep='\t',index=None,header=None)
    test_dataset.to_csv(os.path.join(output_path,'test.tsv'),sep='\t',index=None,header=None)

def addEpitope():
    # 添加epitope列，split cdr3列
    df = pd.read_csv('/aaa/louisyuzhao/project2/immuneDataSet/10x_PMHC/10xPBMC_modified_MaxAtomLenB-336_MaxAtomLenA-415_selected_addAffinityValue.csv')
    epitope_list = []
    for i,row in df.iterrows():
        columns = list(df.columns)
        start = columns.index('A0101_VTEHDTLLY_IE-1_CMV_binder')
        end = columns.index('NR(B0801)_AAKGRGAAL_NC_binder') + 1
        flag = 0
        for j in columns[start:end]:
            if(row[j]==True):
                epitope = j.split('_')[1]
                epitope_list.append(epitope)
                flag = 1
        if(flag ==0):
            epitope_list.append('unknown')
    df['epitope'] = epitope_list
    df.to_csv('/aaa/louisyuzhao/project2/immuneDataSet/10x_PMHC/10xPBMC_modified_MaxAtomLenB-336_MaxAtomLenA-415_selected_addAffinityValue_addEpitope.csv',index=None)

def addSplit():
    df = pd.read_csv('/aaa/louisyuzhao/project2/immuneDataSet/10x_PMHC/10xPBMC_modified_MaxAtomLenB-336_MaxAtomLenA-415_selected_addAffinityValue_addEpitope.csv')
    TRA_cdr3,TRB_cdr3 = [],[]
    a_name = 'TRA_cdr3'
    b_name = 'TRB_cdr3'
    max = 0
    for i,row in df.iterrows():
        a,b ='',''
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
    df['TRA_cdr3_split'] = TRA_cdr3
    df['TRB_cdr3_split'] = TRB_cdr3
    df.to_csv('/aaa/louisyuzhao/project2/immuneDataSet/10x_PMHC/10xPBMC_modified_MaxAtomLenB-336_MaxAtomLenA-415_selected_addAffinityValue_addEpitope_addSplit.csv',index=None)

def splitData():
    # 7:3划分数据集（整样本在train,test集中的比例相同)
    df = pd.read_csv('/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/data/Classification/version_2.0/unique/10x_7epitopes_lightUnique_allData.csv')
    epitopes = ['AVFDRKSDAK', 'ELAGIGILTV', 'GILGFVFTL', 'GLCTLVAML', 'IVTDFSVIK', 'KLGGALQAK', 'RAKFKQLL']
    df = df[['ID','TRA_cdr3_split','TRA_v_gene','TRA_j_gene','TRB_cdr3_split','TRB_v_gene','TRB_j_gene','epitope']]
    # print(df)
    for i in epitopes:
        df_epitope = df[df['epitope'] == i]
        df_other = df[df['epitope'] != i]
        # split train dataset; test dataset

        idx = np.array(range(len(df_epitope)))
        test_size = 0.3
        np.random.shuffle(idx)
        train_idx = np.random.choice(idx, int((1 - test_size) * idx.shape[0]), replace=False)
        test_idx = np.setdiff1d(idx, train_idx)

        
        train_dataset = df_epitope.iloc[train_idx]
        test_dataset = df_epitope.iloc[test_idx]

        idx = np.array(range(len(df_other)))
        test_size = 0.3
        np.random.shuffle(idx)
        train_idx = np.random.choice(idx, int((1 - test_size) * idx.shape[0]), replace=False)
        test_idx = np.setdiff1d(idx, train_idx)

        train_dataset = pd.concat([train_dataset,df_other.iloc[train_idx]],axis=0)
        test_dataset = pd.concat([test_dataset,df_other.iloc[test_idx]],axis=0)

        idx = np.array(range(len(train_dataset)))
        np.random.shuffle(idx)
        train_dataset = train_dataset.iloc[idx]

        idx = np.array(range(len(test_dataset)))
        np.random.shuffle(idx)
        test_dataset = test_dataset.iloc[idx] 

        output_path = '/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/data/Classification/version_2.0/10x_7epitopes_maxlen43'
        
        if not isdir(os.path.join(output_path,i)):
            mkdir(os.path.join(output_path,i))
        train_dataset.to_csv(os.path.join(os.path.join(output_path,i),'train.tsv'),sep='\t',index=None,header=None)
        test_dataset.to_csv(os.path.join(os.path.join(output_path,i),'test.tsv'),sep='\t',index=None,header=None)

def get_train_valid_test(df,valid_size,test_size):
    # 根据valid_size，test_size划分df

    idx = np.array(range(len(df)))
    np.random.shuffle(idx)

    train_idx = np.random.choice(idx, int((1 - test_size - valid_size) * idx.shape[0]), replace=False)
    
    valid_test_idx = np.setdiff1d(idx, train_idx)
    
    p = valid_size/(valid_size + test_size)
    valid_idx = np.random.choice(valid_test_idx, int(p * valid_test_idx.shape[0]), replace=False)

    test_idx = np.setdiff1d(valid_test_idx, valid_idx)

    train_dataset = df.iloc[train_idx]
    valid_dataset = df.iloc[valid_idx]
    test_dataset = df.iloc[test_idx]

    return train_dataset,valid_dataset,test_dataset

# 划分10x数据train_valid_test 作为immuneBlast的分类任务的数据，以及按相同的分法划分DeepTCR的分类数据
def splitData_721(output_path,deeptcr_output_path):
    # 7:2:1划分数据集（整样本在train,test集中的比例相同)
    df = pd.read_csv('/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/data/Classification/version_2.0/unique/10x_7epitopes_lightUnique_allData.csv')
    epitopes = ['AVFDRKSDAK', 'ELAGIGILTV', 'GILGFVFTL', 'GLCTLVAML', 'IVTDFSVIK', 'KLGGALQAK', 'RAKFKQLL']
    # epitopes = ['IVTDFSVIK']
    df = df[['ID','TRA_cdr3_split','TRA_v_gene','TRA_j_gene','TRB_cdr3_split','TRB_v_gene','TRB_j_gene','epitope','TRA_cdr3','TRB_cdr3']]
    # print(df)
    for i in epitopes:
        df_epitope = df[df['epitope'] == i]
        df_other = df[df['epitope'] != i]

        valid_size = 0.2
        test_size = 0.1

        # epitope
        epitope_train_dataset,epitope_valid_dataset,epitope_test_dataset = get_train_valid_test(df_epitope,valid_size,test_size)

        epitope_train_dataset['confirm'] = ['train']*len(epitope_train_dataset)
        epitope_valid_dataset['confirm'] = ['val']*len(epitope_valid_dataset)
        epitope_test_dataset['confirm'] = ['test']*len(epitope_test_dataset)

        
        if not isdir(os.path.join(deeptcr_output_path,i,i)):
            cmd = 'mkdir -p ' + os.path.join(deeptcr_output_path,i,i)
            system(cmd)
        deeptcr_columns = ['ID','TRA_cdr3','TRA_v_gene','TRA_j_gene','TRB_cdr3','TRB_v_gene','TRB_j_gene','confirm']
        epitope_train_dataset[deeptcr_columns].to_csv(os.path.join(os.path.join(deeptcr_output_path,i,i),'train.tsv'),sep='\t',index=None)
        epitope_valid_dataset[deeptcr_columns].to_csv(os.path.join(os.path.join(deeptcr_output_path,i,i),'valid.tsv'),sep='\t',index=None)
        epitope_test_dataset[deeptcr_columns].to_csv(os.path.join(os.path.join(deeptcr_output_path,i,i),'test.tsv'),sep='\t',index=None)
       
        # other
        other_train_dataset,other_valid_dataset,other_test_dataset = get_train_valid_test(df_other,valid_size,test_size)

        other_train_dataset['confirm'] = ['train']*len(other_train_dataset)
        other_valid_dataset['confirm'] = ['val']*len(other_valid_dataset)
        other_test_dataset['confirm'] = ['test']*len(other_test_dataset)

        if not isdir(os.path.join(deeptcr_output_path,i,'unknown')):
            cmd = 'mkdir -p ' + os.path.join(deeptcr_output_path,i,'unknown')
            system(cmd)
        other_train_dataset[deeptcr_columns].to_csv(os.path.join(os.path.join(deeptcr_output_path,i,'unknown'),'train.tsv'),sep='\t',index=None)
        other_valid_dataset[deeptcr_columns].to_csv(os.path.join(os.path.join(deeptcr_output_path,i,'unknown'),'valid.tsv'),sep='\t',index=None)
        other_test_dataset[deeptcr_columns].to_csv(os.path.join(os.path.join(deeptcr_output_path,i,'unknown'),'test.tsv'),sep='\t',index=None)

        train_dataset = pd.concat([epitope_train_dataset, other_train_dataset],axis=0)
        test_dataset = pd.concat([epitope_test_dataset, other_test_dataset],axis=0)
        valid_dataset = pd.concat([epitope_valid_dataset, other_valid_dataset],axis=0)

        idx = np.array(range(len(train_dataset)))
        np.random.shuffle(idx)
        train_dataset = train_dataset.iloc[idx]

        idx = np.array(range(len(test_dataset)))
        np.random.shuffle(idx)
        test_dataset = test_dataset.iloc[idx] 

        idx = np.array(range(len(valid_dataset)))
        np.random.shuffle(idx)
        valid_dataset = valid_dataset.iloc[idx] 

        immuneblast_columns = ['ID','TRA_cdr3_split','TRB_cdr3_split','epitope']
        if not isdir(os.path.join(output_path,i)):
            mkdir(os.path.join(output_path,i))
        train_dataset[immuneblast_columns].to_csv(os.path.join(os.path.join(output_path,i),'train.tsv'),sep='\t',index=None,header=None)
        test_dataset[immuneblast_columns].to_csv(os.path.join(os.path.join(output_path,i),'test.tsv'),sep='\t',index=None,header=None)
        valid_dataset[immuneblast_columns].to_csv(os.path.join(os.path.join(output_path,i),'valid.tsv'),sep='\t',index=None,header=None)

# 根据赵老师的分法，分得VDJ在immuneBlaset分类任务上的数据
def getVDJFromZhao():
    output_path = '/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/data/Classification/version_3.0/VDJ_7_2_1'
    path = '/aaa/louisyuzhao/guy2/xiaonasu/DeepTCR-1.4/Data/DeepAIR/VDJ/zhao'
    dirs = os.listdir(path)
    for file in dirs:
        file_path = os.path.join(path,file)
        df = pd.read_csv(file_path)
        df = df[['ID','TRA_cdr3','TRB_cdr3','epitope']]
        a_name = 'TRA_cdr3'
        b_name = 'TRB_cdr3'
        max = 0
        TRA_cdr3,TRB_cdr3 = [],[]
        for i,row in df.iterrows():
            a,b ='',''
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
        df['TRA_cdr3'] = TRA_cdr3
        df['TRB_cdr3'] = TRB_cdr3
        epitope = file.split("_")[0]
        filename = file.split("_")[1]
        df.to_csv(os.path.join(output_path,epitope,filename+'.tsv'),index=None,sep='\t')

# 从immuneBlast的输入数据，转化为3mer数据
def pase_to_3mer(input_file,output_file):
    df = pd.read_csv(input_file,sep='\t',header=None)
    TRA_cdr3 = []
    TRB_cdr3 = []
    for i,row in df.iterrows():
        a = row[1].split(' ')
        b = row[2].split(' ')
        a_new,b_new = '',''
        for i in range(len(a)-2):
            a_mer = a[i] + a[i+1] + a[i+2]
            a_new += a_mer
            if(i!=len(a)-3):
                a_new += ' '
        for i in range(len(b)-2):
            b_mer = b[i] + b[i+1] + b[i+2]
            b_new += b_mer
            if(i!=len(b)-3):
                b_new += ' '
        TRA_cdr3.append(a_new)
        TRB_cdr3.append(b_new)
    df.insert(1,'TRA_cdr3',TRA_cdr3)
    df.insert(2,'TRB_cdr3',TRB_cdr3)
    df.columns = ['0','1','2','3','4','5']
    df = df[['0','1','2','5']]
    df.to_csv(output_file,index=None,sep='\t',header=None)

# 从ab_3mer转为b_3mer
def process_beta():
    input_path = '/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/data/Classification/version_3.0/10x_3mer_7_2_1'
    output_path = '/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/data/Classification/version_3.0/10x_a_3mer_7_2_1'
    dirs = os.listdir(input_path)
    for p in dirs:
        out_path = os.path.join(output_path,p)
        if not isdir(out_path):
            mkdir(out_path)
        in_path = os.path.join(input_path,p)
        d = os.listdir(in_path)
        for file in d:
            input_file = os.path.join(in_path,file)
            output_file = os.path.join(out_path,file)
            input_df = pd.read_csv(input_file,header=None,sep='\t')
            output_df = input_df.iloc[:,[0,1,3]]
            output_df.to_csv(output_file,index=None,sep='\t',header=None)

# 去除test,valid中在train见过的数据
def dropDuplicates():
    input_path = '/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/data/Classification/version_3.0/10x_b_3mer_7_2_1'
    dirs = os.listdir(input_path)
    for p in dirs:
        in_path = os.path.join(input_path,p)
        d = os.listdir(in_path)
        train = os.path.join(in_path,'train.tsv')
        valid = os.path.join(in_path,'valid.tsv')
        test = os.path.join(in_path,'test.tsv')
        train_df =  pd.read_csv(train,sep='\t',header=None)
        valid_df =  pd.read_csv(valid,sep='\t',header=None)
        test_df =  pd.read_csv(test,sep='\t',header=None)
        train_beta = train_df.iloc[:,1].tolist()

        valid_duplicates = valid_df[valid_df[1].isin(train_beta)]
        valid_df.drop(valid_duplicates.index,inplace=True)
        valid_df.to_csv(valid,index=None,sep='\t',header=None)

        test_duplicates = test_df[test_df[1].isin(train_beta)]
        test_df.drop(test_duplicates.index,inplace=True)
        test_df.to_csv(test,index=None,sep='\t',header=None)
        
def checkLessThan40():
    # 检查数据中是否有大于40的CDR3
    path = '/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/data/Classification/version_3.0'
    dirs = os.listdir(path)
    for d in dirs:
        p = os.path.join(path,d)
        dirs2 = os.listdir(p)
        for d2 in dirs2:
            p2 = os.path.join(p,d2)
            dirs3 = os.listdir(p2)
            for file in dirs3:
                input = os.path.join(p2,file)
                print(input)
                df = pd.read_csv(input,header=None,sep='\t')
                for i,row in df.iterrows():
                    if(len(row[1].split())>40):
                        print(i)
                    if(len(row[2].split())>40):
                        print(i)

def onemer_to_twoMer():
    input_path = '/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/data/Classification/version_3.0/VDJ_normal_7_2_1'
    output_path = '/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/data/Classification/version_3.0/VDJ_2mer_7_2_1'
    dirs = os.listdir(input_path)
    for p in dirs:
        path = os.path.join(input_path,p)
        d = os.listdir(path)
        for file in d:
            input_file = os.path.join(path,file)
            df = pd.read_csv(input_file,sep='\t',header=None)
            two_mer_a = []
            two_mer_b = []
            for i,row in df.iterrows():
                one_mer_a = row[1].split()
                one_mer_b = row[2].split()
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

            df[1] = two_mer_a
            df[2] = two_mer_b
            output = os.path.join(output_path,p)
            print(output)
            if not isdir(output):
                cmd = 'mkdir -p ' + output
                system(cmd)
            df.to_csv(os.path.join(output,file),header=None,index=None,sep='\t')

def to_tcr_bert():
    input_path = '/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/data/Classification/version_3.0/10x_normal_7_2_1'
    output_path = '/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/data/Classification/version_3.0/10x_tcr-bert_7_2_1'
    dirs = os.listdir(input_path)
    for p in dirs:
        path = os.path.join(input_path,p)
        d = os.listdir(path)
        for file in d:
            input_file = os.path.join(path,file)
            df = pd.read_csv(input_file,sep='\t',header=None)
            two_mer_a = []
            two_mer_b = []
            for i,row in df.iterrows():
                one_mer_a = row[1].split()
                one_mer_b = row[2].split()
                # 处理a链
                two_mer = ''
                for j in one_mer_a:
                    two_mer += j
                two_mer_a.append(two_mer)
                # 处理b链
                two_mer = ''
                for j in one_mer_b:
                    two_mer += j
                two_mer_b.append(two_mer)

            df[1] = two_mer_a
            df[2] = two_mer_b
            output = os.path.join(output_path,p)
            print(output)
            if not isdir(output):
                cmd = 'mkdir -p ' + output
                system(cmd)
            df.to_csv(os.path.join(output,file),header=None,index=None,sep='\t')

def add_binding(df,p):
    idx = df[df[3]==p].index.tolist()
    binding = [False]*len(df)
    df['binding'] = binding
    df.loc[idx,'binding'] = True
    return df

def process_BCR(path,deeptcr_output_path,immuneblast_output_path,tcrbert_output_path,disease):

    dirs = os.listdir(path)
    df_tcrbert = pd.DataFrame(columns=['ID','tra','trb','binding','confirm'])
    for file in dirs:
        file_path = os.path.join(path,file)
        df = pd.read_csv(file_path)
        df_deeptcr = df[['ID','TRA_cdr3','TRA_v_gene','TRA_j_gene','TRB_cdr3','TRB_v_gene','TRB_j_gene',disease]]
        df = df[['ID','TRA_cdr3','TRB_cdr3',disease]]
        confirm = file.split('_')[-2]

        # 处理deepTCR数据
        df_deeptcr['confirm'] = [confirm]*len(df_deeptcr)
        df_HIV = df_deeptcr[df_deeptcr[disease] == 1]
        df_unknown = df_deeptcr[df_deeptcr[disease] == 0]
        df_HIV = df_HIV[['ID','TRA_cdr3','TRA_v_gene','TRA_j_gene','TRB_cdr3','TRB_v_gene','TRB_j_gene','confirm']]
        df_unknown = df_unknown[['ID','TRA_cdr3','TRA_v_gene','TRA_j_gene','TRB_cdr3','TRB_v_gene','TRB_j_gene','confirm']]
        if not isdir(os.path.join(deeptcr_output_path,disease)):
            system('mkdir -p ' + os.path.join(deeptcr_output_path,disease))
        if not isdir(os.path.join(deeptcr_output_path,'unknown')):
            system('mkdir -p ' + os.path.join(deeptcr_output_path,'unknown'))
        df_HIV.to_csv(os.path.join(deeptcr_output_path,disease,confirm+'.tsv'),sep='\t',index=None)
        df_unknown.to_csv(os.path.join(deeptcr_output_path,'unknown',confirm+'.tsv'),sep='\t',index=None)

        if(confirm=='val'):
            confirm ='valid'
        # 处理immuneblast数据
        df_immuneblast = df.copy(deep=True)
        TRA_cdr3 = []
        TRB_cdr3 = []
        for i,row in df_immuneblast.iterrows():
            a = list(row[1])
            b = list(row[2])
            a_new,b_new = '',''
            for i in range(len(a)-2):
                a_mer = a[i] + a[i+1] + a[i+2]
                a_new += a_mer
                if(i!=len(a)-3):
                    a_new += ' '
            for i in range(len(b)-2):
                b_mer = b[i] + b[i+1] + b[i+2]
                b_new += b_mer
                if(i!=len(b)-3):
                    b_new += ' '
            TRA_cdr3.append(a_new)
            TRB_cdr3.append(b_new)

        df_immuneblast['TRA_cdr3'] = TRA_cdr3
        df_immuneblast['TRB_cdr3'] = TRB_cdr3
        df_immuneblast.to_csv(os.path.join(immuneblast_output_path,confirm+'.tsv'),sep='\t',header=None,index=None)

        # 处理tcr-bert数据
        df_temp = df.copy(deep=True)
        df_temp['confirm'] = [confirm]*len(df_temp)
        df_temp.columns = ['ID','tra','trb','binding','confirm']
        df_tcrbert = pd.concat([df_temp,df_tcrbert],axis=0)
    df_tcrbert.to_csv(os.path.join(tcrbert_output_path,disease+'.csv'),index=None)

if __name__ == '__main__':

    # addEpitope()
    # addSplit()

    # unique TRA_cdr3,TRB_cdr3去重
    # input_path = '/aaa/louisyuzhao/project2/immuneDataSet/ALL/temp'
    # output_path = '/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/data/Classification/unique/IBD_R_TCR_unique.csv'
    # Unique(input_path=input_path,output_path=output_path)

    # unique_alldata;筛选七个表位的数据
    # input_file = '/aaa/louisyuzhao/project2/immuneDataSet/10x_PMHC/10xPBMC_modified_MaxAtomLenB-336_MaxAtomLenA-415_selected_addAffinityValue_addEpitope_addSplit.csv'
    # df = pd.read_csv(input_file)
    # df.drop_duplicates(['TRA_cdr3','TRB_cdr3','epitope'],inplace=True)
    # epitopes = ['AVFDRKSDAK', 'ELAGIGILTV', 'GILGFVFTL', 'GLCTLVAML', 'IVTDFSVIK', 'KLGGALQAK', 'RAKFKQLL']
    # index = [False]*len(df)
    # for i in epitopes:
    #     index = (index) | (df['epitope'] == i)
    # df = df[index]
    # df.to_csv('/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/data/Classification/version_2.0/unique/10x_7epitopes_lightUnique_allData.csv',index=None)

    # input_file = '/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/data/Classification/unique/IBD_R_BCR_unique.csv'
    # output_path = '/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/data/Classification/IBD/IBD_R_BCR'
    # Split(input_file,output_path)
    
    # 1.处理10x数据：划分10x数据train_valid_test 作为immuneBlast的分类任务的数据，以及按相同的分法划分DeepTCR的分类数据
    # deeptcr_output_path = '/aaa/louisyuzhao/guy2/xiaonasu/test/deeptcr'
    # output_path = '/aaa/louisyuzhao/guy2/xiaonasu/test/immuneblast'
    # splitData_721(output_path,deeptcr_output_path)

    #  2.处理VDJ数据：根据赵老师的分法，分得VDJ在immuneBlaset分类任务上的数据
    # getVDJFromZhao()

    # df = pd.read_csv('/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/result/classification/10x/GILGFVFTL/3mer_1/batch_size64_lr_b0.0001_lr_c0.001/result.csv')
    # real = np.array(df['real'].tolist())
    # pred = np.array(df['pred'].tolist())
    # auc = metrics.roc_auc_score(real,pred)
    # print(auc)

    # input_file = '/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/data/Classification/version_3.0/temp/KLGGALQAK/train.tsv'
    # output_file = '/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/data/Classification/version_3.0/temp_3mer/KLGGALQAK/train.tsv'
    # pase_to_3mer(input_file,output_file)

    # 从ab_3mer转为b_3mer   
    # process_beta()

    # 去除test,valid中在train见过的数据
    # dropDuplicates()

    # 1mer分类数据转为2mer
    # onemer_to_twoMer()

    # 从ab_3mer转为a_3mer   
    # process_beta()

    # 生成tcr-bert数据
    # input_path = '/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/data/Classification/version_3.0/10x_tcr-bert_7_2_1'
    # output_path = '/aaa/louisyuzhao/guy2/xiaonasu/tcr-bert/data/ImmuneBlast/10x'
    # dirs = os.listdir(input_path)
    # for p in dirs:
    #     path = os.path.join(input_path,p)
    #     d = os.listdir(path)
    #     train = df = pd.read_csv(os.path.join(path,'train.tsv'),sep='\t',header=None)
    #     valid = df = pd.read_csv(os.path.join(path,'valid.tsv'),sep='\t',header=None)
    #     test = df = pd.read_csv(os.path.join(path,'test.tsv'),sep='\t',header=None)
    #     train['confirm'] = ['train']*len(train)
    #     train = add_binding(train,p)

    #     valid['confirm'] = ['valid']*len(valid)
    #     valid = add_binding(valid,p)

    #     test['confirm'] = ['test']*len(test)
    #     test = add_binding(test,p)

    #     df = pd.concat([train,test,valid])
    #     df.columns = ['ID','tra','trb','epitope','confirm','binding']
    #     df.to_csv(os.path.join(output_path,p + '.csv'),index=None)

    # 3.处理BCR数据：HIV和Influenza_A
    path = '/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST_finalVersion/rawdata/classification/Plasmodium_seed-15'
    deeptcr_output_path = '/aaa/louisyuzhao/guy2/xiaonasu/DeepTCR-1.4/Data/DeepAIR/Plasmodium_seed-15'
    immuneblast_output_path = '/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST_finalVersion/data/classification/Plasmodium_seed-15'
    tcrbert_output_path = '/aaa/louisyuzhao/guy2/xiaonasu/tcr-bert/data/ImmuneBlast/Plasmodium_seed-15'
    disease = 'Plasmodium'
    process_BCR(path,deeptcr_output_path,immuneblast_output_path,tcrbert_output_path,disease)

 




            
    
        
