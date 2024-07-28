from torch.utils.data import Dataset
from itertools import islice

import tqdm
import torch
import random

import sys
sys.path.append('/aceph/louisyuzhao/buddy2/linyangxiao/SC-AIR/SC-AIR-BERT-Multi/code/bert')
from dataset import BERTDataset_MLM


class DatasetMultimodal(BERTDataset_MLM):
    def __init__(self, 
                corpus_path, 
                vocab, 
                seq_len, 
                class_name, 
                encoding="utf-8", 
                corpus_lines=None, 
                on_memory=True,
                prob=0.10
            ):
        super().__init__(corpus_path, 
                         vocab, 
                         seq_len,
                         encoding, 
                         corpus_lines, 
                         on_memory,prob
                    )
        # self.class_name = class_name
        # if self.on_memory:
            # self.lines = self.lines[1:]  # Skip the first line (header)
        # if not self.on_memory:
            # self.file.readline()

    def __getitem__(self, item):
        ID, TRA_cdr3_3Mer, TRB_cdr3_3Mer, reg, TRA_cdr3, TRA_v_gene, TRA_j_gene, TRB_cdr3, TRB_v_gene, TRB_j_gene = self.get_corpus_line(item)


        t1_random, t1_label = self.random_word(TRA_cdr3_3Mer)
        t2_random, t2_label = self.random_word(TRB_cdr3_3Mer)
        # print('tokens:',t1)
        # print('t1_random:',t1_random)
        # print('t1_label:',t1_label)
        # print('\n')

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1 = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]
        t2 = t2_random + [self.vocab.eos_index]

        t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]
        t2_label = t2_label + [self.vocab.pad_index]

        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]

        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)
        
        # binary
        label = float(reg)

        output = {"ID":int(ID),
                  "bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "reg_label":label
                  }
        
        # print('using dataloader')
        output_tensor = {key: torch.tensor(value) for key, value in output.items()}
        output_tensor["TRA_v_gene"] = TRA_v_gene
        output_tensor["TRA_j_gene"] = TRA_j_gene
        output_tensor["TRB_v_gene"] = TRB_v_gene
        output_tensor["TRB_j_gene"] = TRB_j_gene
        return output_tensor

    def get_corpus_line(self, item):
        if self.on_memory:
            ID, TRA_cdr3_3Mer, TRB_cdr3_3Mer, reg, TRA_cdr3, TRA_v_gene, TRA_j_gene, TRB_cdr3, TRB_v_gene, TRB_j_gene = self.lines[item]
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                # self.file.readline()  # 再次跳过第一行
                line = self.file.__next__()

            ID, TRA_cdr3_3Mer, TRB_cdr3_3Mer, reg, TRA_cdr3, TRA_v_gene, TRA_j_gene, TRB_cdr3, TRB_v_gene, TRB_j_gene = line[:-1].split("\t")

        return ID, TRA_cdr3_3Mer, TRB_cdr3_3Mer, reg, TRA_cdr3, str(TRA_v_gene), str(TRA_j_gene), TRB_cdr3, str(TRB_v_gene), str(TRB_j_gene)



