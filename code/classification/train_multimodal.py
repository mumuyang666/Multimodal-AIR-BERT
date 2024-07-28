import os
from os import system
import os.path
from os.path import isdir
import sys
sys.path.append(os.path.abspath('/aceph/louisyuzhao/buddy2/linyangxiao/SC-AIR/SC-AIR-BERT-Multi/code'))
import time
import torch
from torch.utils.data import DataLoader
from Dataset import DatasetMultimodal
from Dataset_singleSentence import Dataset_singleSentence

from bert.dataset import WordVocab

from FusionModel import FusionModel
from gene.gene_encoder import get_gene_token, save_gene_token, load_saved_token
from gene.gene_encoder import get_gene_encoder

import argparse
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

import nni
import logging
import random

import matplotlib.pyplot as plt

#%%
def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False   

#计算准确率的公式
def binary_accuracy(predict, label):
    rounded_predict = torch.round(predict) #四舍五入
    # print('predict:',predict)
    # print('label:',label)
    correct = (rounded_predict == label).float()
    accuracy = correct.sum() / len(correct)

    return accuracy

#定义训练方法
def train(bert_model, gene_model, model, gene_optimizer, bert_optimizer, optimizer, dataset_loader, train_phase, device):
    #记录统计信息
    epoch_loss, epoch_acc = 0., 0.
    total_len = 0

    total_real = torch.empty(0,dtype=int)
    total_pred = torch.empty(0)
    total_ID = torch.empty(0,dtype=int)
    total_pred.to(device)

    #分batch进行训练
    for i, data in enumerate(dataset_loader):
        if(train_phase):
            if(args.finetune):
                bert_model.train()
            else:
                bert_model.eval()            
            gene_model.train()    
            model.train()        
        else:
            bert_model.eval()
            gene_model.eval() 
            model.eval()
        
        # print(data)
        ID = data['ID']
        label = data['classification_label'].cuda()
        encoding = data['bert_input']
        segment_info = data['segment_label']
        
        # --------------- for debugging --------------
        # for i in range(len(data["TRA_v_gene"])):
        #     print(type(data["TRA_v_gene"][i]))
            
        gene_info_dict = dict()
        gene_info_dict["TRA_v_gene"] = data["TRA_v_gene"]
        gene_info_dict["TRA_j_gene"] = data["TRA_j_gene"]
        gene_info_dict["TRB_v_gene"] = data["TRB_v_gene"]
        gene_info_dict["TRB_j_gene"] = data["TRB_j_gene"]
        gene_df = pd.DataFrame(gene_info_dict)
        # print('gene_df',gene_df)
        gene_info_input = {c:gene_df[c].values for c in gene_info_dict.keys()}
        # --------------- for debugging --------------
        # print(gene_info_input)
        
        # sequence feature
        bert_output = bert_model(encoding.to(device), segment_info.to(device))
        pooler_output = bert_output[:,0,:]
        
        # gene feature 
        tokenized_feature = gene_model.tokenize(gene_info_input)
        tokenized_feature = {key: torch.tensor(value).to(device) for key, value in tokenized_feature.items()} 
        
        gene_output = gene_model(tokenized_feature)
        
        # --------------- for debugging --------------
        # print(pooler_output.size())
        # print(gene_output.size())
        
        Y_prob, Y_hat, features = model(pooler_output, gene_output)
        # --------------- for debugging --------------
        # print(Y_prob)
        Y_prob = Y_prob.squeeze()
        # print(Y_prob)
        # print(label)
        loss = crit(Y_prob, label.float())
        acc = binary_accuracy(Y_prob, label)

        #gd
        if(train_phase):
            optimizer.zero_grad() #把梯度重置为零
            gene_optimizer.zero_grad()
            if(args.finetune):
                bert_optimizer.zero_grad()
            
            loss.backward() #求导
            
            optimizer.step() #更新模型
            gene_optimizer.step()
            if(args.finetune):
                bert_optimizer.step()

        epoch_loss += loss * len(label)
        epoch_acc += acc * len(label)
        total_len += len(label)
        
        total_real = torch.cat([total_real.to(device),label],dim=0)
        total_pred = torch.cat([total_pred.to(device),Y_prob],dim=0)
        total_ID = torch.cat([total_ID,ID],dim=0)

        # print("batch %d loss:%f accuracy:%f" % (i, loss, acc))

    auc = roc_auc_score(total_real.detach().cpu().numpy(), total_pred.detach().cpu().numpy())

    return epoch_loss/total_len, epoch_acc/total_len, auc, total_real.detach().cpu().numpy(), total_pred.detach().cpu().numpy(), total_ID.numpy()

# early stop
def stop_check(loss,stop_criterion,stop_criterion_window):
    w = loss[-stop_criterion_window:]
    # print(w)
    if((w[0]-w[-1])/w[0] < stop_criterion):
        return 1
    else:
        return 0

# plot loss&auc
def plot(Loss_list, Accuracy_list, outputname ,name, x):

    x1 = range(0, len(Loss_list))
    x2 = range(0, len(Accuracy_list))
    y1 = Loss_list
    y2 = Accuracy_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1)

    plt.title(name)
    plt.ylabel('loss')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2)

    plt.xlabel(x)
    plt.ylabel('auc')
    plt.savefig(outputname)
    plt.close()

#%%
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--vocab_path", 
                        required=True, 
                        type=str, 
                        default = os.path.abspath('../../data/vocab/vocab_3mer.pkl'), 
                        help="built vocab model path with bert-vocab"
                    )
    parser.add_argument("-g", "--gene_token", 
                        required=True, 
                        type=str, 
                        default = os.path.abspath('/aceph/louisyuzhao/buddy2/linyangxiao/SC-AIR/SC-AIR-BERT-Multi/data/classification/gene_full.csv'), 
                        help="built vocab model path with bert-vocab"
                    )
    parser.add_argument("-c", "--train_dataset", 
                        required=True, 
                        type=str, 
                        default=os.path.abspath('../../data/classification/10x/10x_ab_3mer_7_2_1/AVFDRKSDAK/train_modified.tsv'), 
                        help="train dataset"
                    )
    parser.add_argument("-d", "--valid_dataset", 
                        required=True, 
                        type=str, 
                        default=os.path.abspath('../../data/classification/10x/10x_ab_3mer_7_2_1/AVFDRKSDAK/valid_modified.tsv'), 
                        help="valid dataset"
                    )
    parser.add_argument("-t", "--test_dataset", 
                        type=str, 
                        default=os.path.abspath('../../data/classification/10x/10x_ab_3mer_7_2_1/AVFDRKSDAK/test_modified.tsv'), 
                        help="test dateset"
                    )
    parser.add_argument("--bert_model", 
                        type=str, 
                        default=os.path.abspath('../../checkpoint/pretrain_models/ab_3mer_len79.ep28'), 
                        help="bert model"
                    )
    parser.add_argument("-o", "--output_path", 
                        required=True, 
                        type=str, 
                        default=os.path.abspath('../../result/classification/test'), 
                        help="ex)output/bert.model"
                    )

    parser.add_argument("-s", "--seq_len", type=int, default=79, help="maximum sequence len")
    parser.add_argument("--prob", type=float, default=0.0, help="prob")

    parser.add_argument("-e", "--epochs", type=int, default=10, help="min epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="number of batch_size")
    parser.add_argument("--lr_a", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument("--lr_b", type=float, default=1e-4, help="learning rate of adam")
    parser.add_argument("--lr_c", type=float, default=1e-3, help="learning rate of adam")

    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--with_cuda", type=bool,  default=True, help="")

    parser.add_argument("--class_name", type=str, default='AVFDRKSDAK', help="class name")
    parser.add_argument("--finetune", type=int, default=1, help="finetune bert")

    parser.add_argument("--chain", type=int, default=2, help="the number of chain")
    parser.add_argument("--seed", type=int, default=27, help="default seed")
    parser.add_argument("--NNI_Search", type=int, default=0, help="NNI Search")
    
    args = parser.parse_args()

    class_name = args.class_name

    # NNI 的参数
    if args.NNI_Search:
        print('Use NNI Search!')
        RCV_CONFIG = nni.get_next_parameter()

        seed = RCV_CONFIG['seed']
        path = os.path.join(args.output_path, class_name,'seed_{}'.format(seed))
    else:
        path = os.path.join(args.output_path,class_name)
        seed = args.seed

    train_dataset = args.train_dataset
    valid_dataset = args.valid_dataset
    test_dataset = args.test_dataset

    setup_seed(seed)

    #载入数据预处理
    print("Loading Vocab")
    vocab = WordVocab.load_vocab(args.vocab_path)
    print("Vocab Size: ", len(vocab))

    print("build gene tokens")
    dataframe = pd.read_csv(os.path.abspath(args.gene_token))
    print(dataframe.head(5))
    gene_tokenizers = get_gene_token(dataframe = dataframe)
    save_gene_token(tokenizers=gene_tokenizers, output_folder=args.output_path)
    
    if(args.chain==1):
        Dataset = Dataset_singleSentence

    print("Loading Train Dataset")
    train_dataset = DatasetMultimodal(train_dataset, 
                                vocab, 
                                seq_len=args.seq_len,
                                on_memory=True,
                                prob = args.prob,
                                class_name = class_name)
    valid_dataset = DatasetMultimodal(valid_dataset, 
                                vocab, 
                                seq_len=args.seq_len,
                                on_memory=True,
                                prob = args.prob,
                                class_name = class_name)
    test_dataset = DatasetMultimodal(test_dataset, 
                                vocab, 
                                seq_len=args.seq_len,
                                on_memory=True,
                                prob = args.prob,
                                class_name = class_name)
    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=32, shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=32)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=32)
    print("数据载入完成")

    # #设置运行设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("设备配置完成")

    # load bert model
    bert_model = torch.load(args.bert_model)
    bert_model = bert_model.to(device)
    print("bert层模型创建完成")

    # load gene model
    gene_model = get_gene_encoder(gene_tokenizer = gene_tokenizers)
    gene_model = gene_model.to(device)
    print("gene层模型创建完成")

    # #创建模型对象
    networkOption = dict()
    networkOption['fusion_type']='pofusion'
    networkOption['skip']= True
    networkOption['use_bilinear']= True
    networkOption['input1_gate']= True
    networkOption['input2_gate']= True
    networkOption['input1_dim']= 512
    networkOption['input2_dim']= 48
    networkOption['input1_scale']= 4 
    networkOption['input2_scale']= 1
    networkOption['mmhid']=64
    networkOption['dropout_rate']=0.25
    networkOption['label_dim'] = 1
    networkOption['activation'] ='Sigmoid'
    model = FusionModel(networkOption)
    model = model.to(device)
    print("全连接层模型创建完成")

    # if args.with_cuda and torch.cuda.device_count() > 1:
    #     print("Using %d GPUS" % torch.cuda.device_count())
    #     bert_model = torch.nn.DataParallel(bert_model, device_ids=args.cuda_devices)
    #     gene_model = torch.nn.DataParallel(gene_model, device_ids=args.cuda_devices)
    #     model = torch.nn.DataParallel(model, device_ids=args.cuda_devices)

    #定义优化器&损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_c)
    gene_optimizer = torch.optim.Adam(gene_model.parameters(), lr=args.lr_a)
    bert_optimizer = torch.optim.Adam(bert_model.parameters(), lr=args.lr_b)
    # if(args.finetune):
    #     # 半冻结
    #     unfreeze_layers = ['embedding','transformer_blocks.4','transformer_blocks.5']
    
    #     for name, param in bert_model.named_parameters():
    #         print(name,param.size())

    #     for name ,param in bert_model.named_parameters():
    #         param.requires_grad = False
    #         for ele in unfreeze_layers:
    #             if ele in name:
    #                 param.requires_grad = True
    #                 break
    #     #验证一下
    #     print("验证")
    #     for name, param in bert_model.named_parameters():
    #         if param.requires_grad:
    #             print(name,param.size())
    
    #     #过滤掉requires_grad = False的参数
    #     bert_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, bert_model.parameters()), lr=args.lr_b)
    crit = torch.nn.BCELoss()
  
    #开始训练
    print("Training Start")
    index = 0

    train_loss_total,val_loss_total = [],[]
    stop_check_list = []
    epoch = 0
    epochs_min = 10

    min_loss = 100
    max_auc = 0
    min_loss_auc,min_loss_acc = 0,0
    max_auc_auc,max_auc_acc = 0,0
    last_epoch_auc,last_epoch_acc = 0,0

    epoch_train_loss_list = []
    epoch_train_auc_list = []
    epoch_valid_loss_list = []
    epoch_valid_auc_list = []
    epoch_test_loss_list = []
    epoch_test_auc_list = []

    #创建输出目录
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    while(True):
        start_time = time.time() 
        epoch_train_loss, epoch_train_acc,epoch_train_auc, epoch_train_real, epoch_train_pred, epoch_train_ID = train(bert_model, 
                                                                                                                    gene_model, 
                                                                                                                    model, 
                                                                                                                    gene_optimizer,
                                                                                                                    bert_optimizer,
                                                                                                                    optimizer,
                                                                                                                    dataset_loader = train_data_loader, 
                                                                                                                    train_phase = True, 
                                                                                                                    device = device
                                                                                                            )
        epoch_train_loss_list.append(epoch_train_loss)
        epoch_train_auc_list.append(epoch_train_auc)
        index += 1
        print("EPOCH %d_train loss:%f accuracy:%f auc:%f" % (index, epoch_train_loss, epoch_train_acc, epoch_train_auc))
        
        train_loss_total.append(epoch_train_loss)
        stop_criterion = 0.001
        stop_criterion_window = 10

        with torch.no_grad():
            epoch_valid_loss, epoch_valid_acc,epoch_valid_auc,epoch_valid_real, epoch_valid_pred, epoch_valid_ID = train(bert_model, 
                                                                                                                        gene_model, 
                                                                                                                        model, 
                                                                                                                        gene_optimizer,
                                                                                                                        bert_optimizer,
                                                                                                                        optimizer,
                                                                                                                        dataset_loader = valid_data_loader,
                                                                                                                        train_phase=False,
                                                                                                                        device = device
                                                                                                                        )
            epoch_valid_loss_list.append(epoch_valid_loss)
            epoch_valid_auc_list.append(epoch_valid_auc)
            print("EPOCH %d_valid loss:%f accuracy:%f auc:%f" % (index, epoch_valid_loss, epoch_valid_acc, epoch_valid_auc))
            
            val_loss_total.append(epoch_valid_loss)

            epoch_test_loss, epoch_test_acc,epoch_test_auc,epoch_test_real, epoch_test_pred, epoch_test_ID = train(bert_model, 
                                                                                                                    gene_model, 
                                                                                                                    model, 
                                                                                                                    gene_optimizer,
                                                                                                                    bert_optimizer,
                                                                                                                    optimizer,
                                                                                                                    dataset_loader = test_data_loader,
                                                                                                                    train_phase=False,
                                                                                                                    device = device
                                                                                                                )
            epoch_test_loss_list.append(epoch_test_loss)
            epoch_test_auc_list.append(epoch_test_auc)
            print("EPOCH %d_test loss:%f accuracy:%f auc:%f" % (index, epoch_test_loss, epoch_test_acc,epoch_test_auc))
            end_time = time.time()  # 记录epoch结束的时间
            epoch_time = end_time - start_time  # 计算epoch的运行时间
            print("Time for this epoch: %.2f seconds" % epoch_time)  #
            
            # 保存valid_loss最小的模型和结果
            if(min_loss > epoch_valid_loss):
                min_loss = epoch_valid_loss
                min_loss_auc = epoch_test_auc
                min_loss_acc = epoch_test_acc
                # 保存模型
                model_output = os.path.join(path,'min_loss_model.pth')
                state = {
                    'bert_model':bert_model.state_dict(),
                    'fc_model':model.state_dict()
                }
                torch.save(state, model_output)
                # model.to(device)
                # 保存预测结果
                data = {
                    'ID':epoch_test_ID,
                    'real':epoch_test_real.tolist(),
                    'pred':epoch_test_pred.tolist()
                }
                df = pd.DataFrame(data)
                df.to_csv(os.path.join(path,'min_loss_result.csv'),index=None)

            # 保存valid_auc最大的模型和结果
            if(max_auc<epoch_valid_auc):
                max_auc = epoch_valid_auc
                max_auc_auc = epoch_test_auc
                max_auc_acc = epoch_test_acc
                model_output = os.path.join(path,'max_auc_model.pth')
                state = {
                    'bert_model':bert_model.state_dict(),
                    'fc_model':model.state_dict()
                }
                torch.save(state, model_output)
                # model.to(device)
                # 保存预测结果
                data = {
                    'ID':epoch_test_ID,
                    'real':epoch_test_real.tolist(),
                    'pred':epoch_test_pred.tolist()
                }
                df = pd.DataFrame(data)
                df.to_csv(os.path.join(path,'max_auc_result.csv'),index=None)
                
            # 删除注释
            if epoch > epochs_min:
                if val_loss_total:
                    stop_check_list.append(stop_check(val_loss_total, stop_criterion, stop_criterion_window))
                    if np.sum(stop_check_list[-3:]) >= 3:
                        # 保存最后一个epochs的结果
                        last_epoch_auc = epoch_test_auc
                        last_epoch_acc = epoch_test_acc
                        model_output = os.path.join(path,'last_epoch_model.pth')
                        state = {
                            'bert_model':bert_model.state_dict(),
                            'fc_model':model.state_dict()
                        }
                        torch.save(state, model_output)
                        # model.to(device)
                        # 保存预测结果
                        data = {
                            'ID':epoch_test_ID,
                            'real':epoch_test_real.tolist(),
                            'pred':epoch_test_pred.tolist()
                        }
                        df = pd.DataFrame(data)
                        df.to_csv(os.path.join(path,'last_epoch_result.csv'),index=None)
                        break
                
            # if(epoch==5):
            #     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_c*0.1)
            #     bert_optimizer = torch.optim.Adam(bert_model.parameters(), lr=args.lr_b*0.1)


        # if(epoch>29):
        #     break
        epoch += 1

    auc_csv = pd.DataFrame(columns=['lr_b','lr_c'])
    auc_csv['max_auc_auc'] = [max_auc_auc]
    auc_csv['max_auc_acc'] = [max_auc_acc.item()]

    auc_csv['min_loss_auc'] = [min_loss_auc]
    auc_csv['min_loss_acc'] = [min_loss_acc.item()]

    auc_csv['last_epoch_auc'] = [last_epoch_auc]
    auc_csv['last_epoch_acc'] = [last_epoch_acc.item()]

    auc_csv['bert_model'] = args.bert_model
    auc_csv['finetune'] = args.finetune
    auc_csv['prob'] = args.prob
    auc_csv['seq_len'] = args.seq_len
    auc_csv['batch_size'] = args.batch_size
    auc_csv['lr_b'] = args.lr_b
    auc_csv['lr_c'] = args.lr_c

    auc_csv.to_csv(os.path.join(path,'parameters.csv'),index=None)

    nni.report_final_result(min_loss_auc)

    # plot(epoch_train_loss_list, 
    #      epoch_train_auc_list,
    #      os.path.join(path,'train_loss_auc.png'),
    #      class_name + '_train','epochs'
    #      )
    # plot(epoch_valid_loss_list,
    #      epoch_valid_auc_list,
    #      os.path.join(path,'valid_loss_auc.png'),
    #      class_name + '_valid','epochs'
    #      )
    # plot(epoch_test_loss_list,
    #      epoch_test_auc_list,
    #      os.path.join(path,'test_loss_auc.png'),
    #      class_name + '_test','epochs'
    #      )

    print('auc:',min_loss_auc)
    print('acc:',min_loss_acc.item())