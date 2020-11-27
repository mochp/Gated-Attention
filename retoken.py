import csv
import torch
import numpy as np
from transformers import BertTokenizer, BertModel

path = '/home/nlp/Gated-Attention/dataset/stsa_seg_bert/test.csv'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


f = csv.reader(open(path,'r'))



bert_token = []
for i,txt in enumerate(f):
    add = []
    label = txt[0]
    text = txt[1]
    text_list = txt[1].split(" ")
    cut = str(txt[2]).split(" ")
    cut = [int(x) for x in cut]
    if len(cut)>8:
        #多于8，删除
        dels = len(cut)-8
        cut = cut[:dels]+cut[dels*2:]

    for t in range(len(cut)-1):
        split = text_list[int(cut[t]):int(cut[t+1])]
        len_text = len(split)
        len_token = len(tokenizer.tokenize(" ".join(split)))
        add.append(len_token-len_text)
    
    new_cut = [0]
    for t in range(1,len(cut)):
        new_cut.append(int(cut[t])+sum(add[:t]))

    # 补充2
    new_cut =[x+1 for x in new_cut]
    new_cut[-1] = new_cut[-1]+1
    new_cut[0] = new_cut[0]-1
    str_new_cut = " ".join([str(x) for x in new_cut])
    bert_token.append([txt[0],txt[1],str_new_cut])


with open("test.csv",'a') as f:
    for i,txt in enumerate(bert_token):
        print(txt)
        write=csv.writer(f)
        write.writerow(txt)