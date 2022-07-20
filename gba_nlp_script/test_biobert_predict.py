# import the whole modules
from cmath import log
import os
import time
start_time = time.perf_counter()
from re import X
import sys
import pickle
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset,DataLoader
from typing import List, Tuple, Dict, Any, Sequence, Optional, Union
from transformers import BertTokenizer, BertModel
from transformers import logging as trans_logging
#log info
logger = logging.getLogger(__name__)
trans_logging.set_verbosity_error()
#logging.basicConfig(level=logging.WARNING)
#some config settings for subsequent prediction
class Config(object):
    model_name = 'lm'  # ['cnn', 'gcn', 'lm']
    use_pcnn = True 
    dropout = 0.3
    num_relations = 31
    #lm_file = 'bert-base-chinese'
    lm_file = '/home/liwenqing/liwenqing_hpc/2_software/DeepKE/pretrained/3_biobert-v1.1'
    lm_num_hidden_layers = 2
    rnn_layers = 2  
cfg = Config()

# %%
Path = str
#load relation csv 
def load_relation_csv(fp:Path) -> List:
    tmp_list = []
    with open(fp, encoding='utf-8') as f:
        for line in f.readlines()[1:]:
            head_type,tail_type,relation,index = line.rstrip("\n").rsplit(",",maxsplit=3)
            tmp_dict = {"head_type":head_type,"tail_type":tail_type,"index":index,"relation":relation}
            tmp_list.append(tmp_dict)
        #reader = csv.DictReader(f, dialect=dialect)
        #return list(reader)
        return tmp_list        
#transform relation to index
def _handle_relation_data(relation_data: List[Dict]) -> Dict:
    rels = dict()
    for d in relation_data:
        rels[d['relation']] = {
            'index': int(d['index']),
            'head_type': d['head_type'],
            'tail_type': d['tail_type'],
        }
    return rels
#transform word to ids using bertTokenier 
def _lm_serialize(data: List[Dict], cfg):
    logger.info('use bert tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(cfg.lm_file)
    for d in data:
        #encode sentence
        sent = d['sentence'].strip()
        sent = sent.replace(d['head'], d['head_type'], 1).replace(d['tail'], d['tail_type'], 1)
        #add [SEP] for latter training
        sent += '[SEP]' + d['head'] + '[SEP]' + d['tail']
        #transform word to ids
        d['token2idx'] = tokenizer.encode(sent, add_special_tokens=True)
        d['lens'] = len(d['token2idx'])

# pytorch construct Dataset
def collate_fn(cfg):
    def collate_fn_intra(batch):
        batch.sort(key=lambda data: int(data['lens']), reverse=True)
        max_len = int(batch[0]['lens'])
        
        def _padding(x, max_len):
            return x + [0] * (max_len - len(x))
        
        def _pad_adj(adj, max_len):
            adj = np.array(adj)
            pad_len = max_len - adj.shape[0]
            for i in range(pad_len):
                adj = np.insert(adj, adj.shape[-1], 0, axis=1)
            for i in range(pad_len):
                adj = np.insert(adj, adj.shape[0], 0, axis=0)
            return adj
        
        #x, y = dict(), []
        x = dict()
        word, word_len = [], []
        head_pos, tail_pos = [], []
        pcnn_mask = []
        adj_matrix = []
        sentences = []
        for data in batch:
            sentences.append(data["text"])
            word.append(_padding(data['token2idx'], max_len))
            word_len.append(int(data['lens']))
            #y.append(int(data['rel2idx']))
            
            if cfg.model_name != 'lm':
                head_pos.append(_padding(data['head_pos'], max_len))
                tail_pos.append(_padding(data['tail_pos'], max_len))
                if cfg.model_name == 'gcn':
                    head = eval(data['dependency'])
                    adj = head_to_adj(head, directed=True, self_loop=True)
                    adj_matrix.append(_pad_adj(adj, max_len))

                if cfg.use_pcnn:
                    pcnn_mask.append(_padding(data['entities_pos'], max_len))

        x['word'] = torch.tensor(word)
        x['lens'] = torch.tensor(word_len)
        x["text"] = sentences
        #y = torch.tensor(y)
        
        if cfg.model_name != 'lm':
            x['head_pos'] = torch.tensor(head_pos)
            x['tail_pos'] = torch.tensor(tail_pos)
            if cfg.model_name == 'gcn':
                x['adj'] = torch.tensor(adj_matrix)
            if cfg.model_name == 'cnn' and cfg.use_pcnn:
                x['pcnn_mask'] = torch.tensor(pcnn_mask)
        return x
        #return x, y
    
    return collate_fn_intra
#return dataset
class CustomDataset(Dataset):
    def __init__(self, Data):
        self.file = Data

    def __getitem__(self, item):
        sample = self.file[item]
        return sample

    def __len__(self):
        return len(self.file)
# pretrain language model
class PretrainLM(nn.Module):
    def __init__(self, cfg):
        #super pytorch model
        super(PretrainLM, self).__init__()
        self.num_layers = cfg.rnn_layers
        self.lm = BertModel.from_pretrained(cfg.lm_file, num_hidden_layers=cfg.lm_num_hidden_layers)
        self.bilstm = nn.LSTM(768,10,batch_first=True,bidirectional=True,num_layers=cfg.rnn_layers,dropout=cfg.dropout)
        self.fc = nn.Linear(20, cfg.num_relations)
    #forward-propagation
    def forward(self, x):
        N = self.num_layers
        word, lens = x['word'], x['lens']
        B = word.size(0)
        #output, pooler_output = self.lm(word)
        output = self.lm(word).last_hidden_state
        output = pack_padded_sequence(output, lens, batch_first=True, enforce_sorted=True)
        _, (output,_) = self.bilstm(output)
        output = output.view(N, 2, B, 10).transpose(1, 2).contiguous().view(N, B, 20).transpose(0, 1)
        output = output[:,-1,:]
        output = self.fc(output)
        
        return output

#loading relation csv
relation_fp = os.path.join('./semmedb_data/origin/relation.csv')
#loading relation data to dict
relation_data = load_relation_csv(relation_fp)
#transform data to bert ids
rels = _handle_relation_data(relation_data)
relation_dict = {}
for label in rels:
    idx = rels[label]["index"]
    relation_dict[idx] = label
#/share/home/liwenqing/liwenqing_hpc/5_neo4j_3.5/4_deepke_paper/kg_pubmed_abstracts.csv.ner
predict_sents = sys.argv[1]
output_predicted_sents = predict_sents + ".predict"
sents_list = []
#reading ner file
with open(predict_sents,"r") as f:
    for line in f.readlines():
        idx,sents = line.rstrip("\n").split("|",maxsplit=1)
        text,head_preferred_name,head_mention,head_type,head_offset,head_id,tail_preferred_name,\
            tail_mention,tail_type,tail_offset,tail_id = sents.rsplit("|",maxsplit=10)
        sents_dict = {"text":sents,"sentence":text,"head":head_mention,"head_offset":head_offset,\
            "tail":tail_mention,"tail_offset":tail_offset,"head_type":head_type,"tail_type":tail_type}
        sents_list.append(sents_dict)
#sents serialization
_lm_serialize(sents_list,cfg)

#pytorch dataloader
train_dataset = CustomDataset(sents_list)
train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn(cfg))

#pretrained bert model
model_path = "./semmedb_data/out/params_biobert_model_25_epoch.pkl"
model = PretrainLM(cfg)
#load model parameters
model.load_state_dict(torch.load(model_path))
#eval mode
model.eval()
#from cuda to cpu
model.cpu()

with open(output_predicted_sents,"w") as o:
    for x in train_dataloader:
        pred_y = model(x).detach().numpy()
        #return argmax index
        pred_y_idx = np.argmax(pred_y,axis=-1)[0]
        o.write("\t".join([x["text"][0],relation_dict[pred_y_idx]])+"\n")
end_time = time.perf_counter()
print("time used:%.1fs"%(end_time-start_time))
