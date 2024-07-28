import torch
from torch.utils.data import DataLoader
from FCModel import FCModel
from Dataset import Dataset

import sys
sys.path.append('..')
import model
from dataset import BERTDataset_MLM, WordVocab

import argparse
from sklearn.metrics import roc_auc_score
import numpy as np
import os.path
import pandas as pd
import os
from os.path import isdir
from os import system

import nni
import logging

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument("-v", "--vocab_path", required=True, type=str, help="built vocab model path with bert-vocab")
parser.add_argument("-i", "--input_path", type=str, default='', help="input_path")
parser.add_argument("-o", "--output_path", required=True, type=str, help="ex)output/bert.model")

parser.add_argument("-s", "--seq_len", type=int, default=60, help="maximum sequence len")
parser.add_argument("--prob", type=float, default=0.10, help="prob")

parser.add_argument("-b", "--batch_size", type=int, default=64, help="number of batch_size")
parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
parser.add_argument("--lr_b", type=float, default=1e-4, help="learning rate of adam")
parser.add_argument("--lr_c", type=float, default=1e-3, help="learning rate of adam")

parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
parser.add_argument("--with_cuda", type=bool,  default=True, help="")
# new
parser.add_argument("--class_name", type=str, default=None, help="class name")
parser.add_argument("--bert_model", type=str, default=None, help="bert model")
parser.add_argument("--finetune", type=int, default=1, help="finetune bert")

parser.add_argument("--NNI_Search", type=int, default=1, help="NNI Search")
parser.add_argument("--in_features", type=int, default=256, help="NNI Search")
args = parser.parse_args()

# NNI 的参数
if args.NNI_Search:
    print('Use NNI Search!')
    RCV_CONFIG = nni.get_next_parameter()
    
    input_path = RCV_CONFIG['input_path']
    class_name = args.class_name
    train_dataset = os.path.join(input_path,'train.tsv')
    test_dataset = os.path.join(input_path,'test.tsv')
    output_path = args.output_path

    # bert_model_path = RCV_CONFIG['bert_model']
    bert_model_path = args.bert_model
    prob = args.prob
    finetune = args.finetune

    batch_size = RCV_CONFIG['batch_size']
    # batch_size = args.batch_size
    lr_b = RCV_CONFIG['lr_b']
    lr_c = RCV_CONFIG['lr_c']

else:
    input_path = args.input_path
    class_name = args.class_name
    train_dataset = os.path.join(input_path,'train.tsv')
    test_dataset = os.path.join(input_path,'test.tsv')
    output_path = args.output_path

    bert_model_path = args.bert_model
    prob = args.prob
    finetune = args.finetune

    batch_size = args.batch_size
    lr_b = args.lr_b
    lr_c = args.lr_c



#载入数据预处理
print("Loading Vocab")
vocab = WordVocab.load_vocab(args.vocab_path)
print("Vocab Size: ", len(vocab))

print("Loading Train Dataset")
train_dataset = Dataset(train_dataset, 
                            vocab, 
                            seq_len=args.seq_len,
                            on_memory=True,
                            prob = prob,
                            class_name = class_name)
test_dataset = Dataset(test_dataset, 
                            vocab, 
                            seq_len=args.seq_len,
                            on_memory=True,
                            prob = prob,
                            class_name = class_name)
print("Creating Dataloader")
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=32,shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=32)
print("数据载入完成")

# #设置运行设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("设备配置完成")

# #加载bert模型
# all---
# /aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST_all_NNI/result/NNI/top_epochs150_lr0.001_hidden256_layers2_batchsize64_prob0.1/bert.model.ep1
# 10x
# /aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/result/NNI/all_cdr3a_cdr3b/top1_epochs100_lr0.001_hidden256_layers2_batchsize32/bert.model.ep99
# 10x_IEDB
# /aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/result/NNI/10x_IEDB_ab/epochs150_lr0.001_hidden256_layers8_batchsize64_prob0.15/bert.model.ep18

bert_model = torch.load(bert_model_path)
bert_model.to(device)
print("bert层模型创建完成")

# #创建模型对象
model = FCModel(in_features=args.in_features)
model = model.to(device)
print("全连接层模型创建完成")

if args.with_cuda and torch.cuda.device_count() > 1:
    print("Using %d GPUS" % torch.cuda.device_count())
    bert_model = torch.nn.DataParallel(bert_model, device_ids=args.cuda_devices)
    model = torch.nn.DataParallel(model, device_ids=args.cuda_devices)

#定义优化器&损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=lr_c)
if(finetune):
    bert_optimizer = torch.optim.Adam(bert_model.parameters(), lr=lr_b)
crit = torch.nn.BCELoss()

#计算准确率的公式
def binary_accuracy(predict, label):
    rounded_predict = torch.round(predict) #四舍五入
    # print('predict:',predict)
    # print('label:',label)
    correct = (rounded_predict == label).float()
    accuracy = correct.sum() / len(correct)

    return accuracy

#定义训练方法
def train(dataset_loader,train):
    #记录统计信息
    epoch_loss, epoch_acc = 0., 0.
    total_len = 0

    total_real = torch.empty(0,dtype=int)
    total_pred = torch.empty(0)
    total_ID = torch.empty(0,dtype=int)
    total_pred.to(device)
    total_real.to(device)

    #分batch进行训练
    for i, data in enumerate(dataset_loader):
        if(train):
            if(finetune):
                bert_model.train()
            else:
                bert_model.eval()
            model.train()
        else:
            bert_model.eval()
            model.eval()

        
        label = data['classification_label']
        label = label.cuda()

        encoding = data['bert_input']
        segment_info = data['segment_label']
        ID = data['ID']
        bert_output = bert_model(encoding.to(device),segment_info.to(device))
        # print('bert_output:',bert_output.shape)
 
        pooler_output = bert_output[:,0,:]
        # print('pooler_output:',pooler_output.shape)
        
        predict = model(pooler_output).squeeze()
        loss = crit(predict, label.float())
        acc = binary_accuracy(predict, label)

        #gd
        if(train):
            
            optimizer.zero_grad() #把梯度重置为零
            if(finetune):
                bert_optimizer.zero_grad()
            loss.backward() #求导
            
            optimizer.step() #更新模型
            if(finetune):
                bert_optimizer.step()

        epoch_loss += loss * len(label)
        epoch_acc += acc * len(label)
        total_len += len(label)
        
        total_real = torch.cat([total_real.to(device),label],dim=0)
        total_pred = torch.cat([total_pred.to(device),predict],dim=0)
        total_ID = torch.cat([total_ID,ID],dim=0)

        # print("batch %d loss:%f accuracy:%f" % (i, loss, acc))

    auc = roc_auc_score(total_real.detach().cpu().numpy(),total_pred.detach().cpu().numpy())

    return epoch_loss/total_len, epoch_acc/total_len, auc, total_real.detach().cpu().numpy(), total_pred.detach().cpu().numpy(), total_ID.numpy()


def stop_check(loss,stop_criterion,stop_criterion_window):
    w = loss[-stop_criterion_window:]
    if((w[0]-w[-1])/w[0] < stop_criterion):
        return 1
    else:
        return 0

def plot(Loss_list,Accuracy_list,outputname,name,x):

    x1 = range(0, len(Loss_list))
    x2 = range(0, len(Accuracy_list))
    y1 = Loss_list
    y2 = Accuracy_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1)
    #     plt.plot(x1, y1, 'o-')
    plt.title(name)
    plt.ylabel('loss')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2)
    #  plt.plot(x2, y2, '.-')
    plt.xlabel(x)
    plt.ylabel('auc')
    plt.savefig(outputname)
    plt.close()

#开始训练
print("Training Start")
Num_Epoch = args.epochs
index = 0

test_loss_total = []
stop_check_list = []
epoch = 0
epochs_min = 10

epoch_train_loss_list = []
epoch_train_auc_list = []
epoch_test_loss_list = []
epoch_test_auc_list = []

max_auc = 0
max_acc = 0

path = os.path.join(output_path, input_path.split('/')[-1],'lr_b' + str(lr_b) + '_lr_c' + str(lr_c))
if not(isdir(path)):
    cmd = 'mkdir -p ' + path
    system(cmd)

while(True):
    epoch_train_loss, epoch_train_acc,epoch_train_auc, epoch_train_real, epoch_train_pred,epoch_train_ID = train(train_data_loader,train=True)
    epoch_train_loss_list.append(epoch_train_loss)
    epoch_train_auc_list.append(epoch_train_auc)
    index += 1
    print("EPOCH %d_train loss:%f accuracy:%f auc:%f" % (index, epoch_train_loss, epoch_train_acc, epoch_train_auc))
    
    stop_criterion = 0.001
    stop_criterion_window = 10

    with torch.no_grad():
        epoch_test_loss, epoch_test_acc,epoch_test_auc,epoch_test_real, epoch_test_pred, epoch_test_ID = train(test_data_loader,train=False)
        epoch_test_loss_list.append(epoch_test_loss)
        epoch_test_auc_list.append(epoch_test_auc)
        print("EPOCH %d_test loss:%f accuracy:%f auc:%f" % (index, epoch_test_loss, epoch_test_acc,epoch_test_auc))
    
        test_loss_total.append(epoch_test_loss)

        # 保存test_auc最高的模型和结果
        if(epoch_test_auc>max_auc):
            max_auc = epoch_test_auc
            max_acc = epoch_test_acc
            # 保存模型
            model_output = os.path.join(path,'model.pth')
            state = {
                'bert_model':bert_model.state_dict(),
                'fc_model':model.state_dict()
            }
            torch.save(state, model_output)
            data = {
                'ID':epoch_test_ID,
                'real':epoch_test_real.tolist(),
                'pred':epoch_test_pred.tolist()
            }
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(path,'result.csv'),index=None)

        # if epoch > epochs_min:
        #     if test_loss_total:
        #         stop_check_list.append(stop_check(test_loss_total, stop_criterion, stop_criterion_window))
        #         if np.sum(stop_check_list[-3:]) >= 3:
        #             break
        
    if(epoch>=99):
        break
    epoch += 1

auc_csv = pd.DataFrame(columns=['auc','acc'])
auc_csv['auc'] = [max_auc]
auc_csv['acc'] = [max_acc.item()]

auc_csv['bert_model'] = bert_model_path
auc_csv['finetune'] = finetune
auc_csv['prob'] = prob
auc_csv['seq_len'] = args.seq_len
auc_csv['batch_size'] = batch_size
auc_csv['lr_b'] = lr_b
auc_csv['lr_c'] = lr_c

auc_csv.to_csv(os.path.join(path,'parameters.csv'),index=None)

nni.report_final_result(max_auc)

plot(epoch_train_loss_list,epoch_train_auc_list,os.path.join(path,'train_loss_auc.png'),class_name + '_train','epochs')
plot(epoch_test_loss_list,epoch_test_auc_list,os.path.join(path,'test_loss_auc.png'),class_name + '_test','epochs')

print('auc:',max_auc)
print('acc:',max_acc)




