import sys
sys.path.append('/aaa/louisyuzhao/project1/SC-AIR-BERT/code/bert')
from dataset import WordVocab
import pickle as pk

if __name__ == "__main__":
    input_file = '/aaa/louisyuzhao/project1/SC-AIR-BERT/data/vocab/vocab_3mer.pkl'
    file = open(input_file, 'rb')
    input_dict = pk.load(file)
    file.close()
    print(input_dict)