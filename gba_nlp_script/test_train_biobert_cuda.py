# import the whole modules
import os
import csv
#loading module
import math
import pickle
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import precision_recall_fscore_support
from typing import List, Tuple, Dict, Any, Sequence, Optional, Union
from transformers import BertTokenizer, BertModel

logger = logging.getLogger(__name__)


# Configuration file of model parameters
class Config(object):
    model_name = 'lm'  # ['cnn', 'gcn', 'lm']
    use_pcnn = True
    min_freq = 1
    pos_limit = 20
    out_path = 'semmedb_data/out'   
    batch_size = 32
    word_dim = 10
    pos_dim = 5
    dim_strategy = 'sum'  # ['sum', 'cat']
    out_channels = 20
    intermediate = 10
    kernel_sizes = [3, 5, 7]
    activation = 'gelu'
    pooling_strategy = 'max'
    dropout = 0.3
    epoch = 30
    num_relations = 20
    learning_rate = 2e-5
    lr_factor = 0.7 # 学习率的衰减率
    lr_patience = 3 # 学习率衰减的等待epoch
    weight_decay = 1e-3 # L2正则
    early_stopping_patience = 6
    train_log = True
    log_interval = 1
    show_plot = True
    only_comparison_plot = False
    plot_utils = 'matplot'
    #lm_file = 'bert-base-chinese'
    lm_file = '/home/liwenqing/liwenqing_hpc/2_software/DeepKE/pretrained/3_biobert-v1.1/'
    lm_num_hidden_layers = 2
    rnn_layers = 2
    
cfg = Config()

# Functions required for preprocessing
Path = str

#load text file
def load_csv(fp: Path, is_tsv: bool = False, verbose: bool = True) -> List:
    #if verbose:
    #   logger.info(f'load csv from {fp}')

    #dialect = 'excel-tab' if is_tsv else 'excel'
    tmp_list = []
    with open(fp, encoding='utf-8') as f:
        for line in f.readlines()[1:]:
            sentence,relation,head,head_offset,tail,tail_offset = line.rstrip("\n").rsplit(",",maxsplit=5)
            tmp_dict = {"sentence":sentence,"relation":relation,"head":head,"head_offset":head_offset,"tail":tail,"tail_offset":tail_offset}
            tmp_list.append(tmp_dict)
        #reader = csv.DictReader(f, dialect=dialect)
        #return list(reader)
        return tmp_list

#load relation csv file   
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

#
def load_pkl(fp: Path, verbose: bool = True) -> Any:
    if verbose:
        logger.info(f'load data from {fp}')

    with open(fp, 'rb') as f:
        data = pickle.load(f)
        return data


def save_pkl(data: Any, fp: Path, verbose: bool = True) -> None:
    if verbose:
        logger.info(f'save data in {fp}')

    with open(fp, 'wb') as f:
        pickle.dump(data, f)
    
    
def _handle_relation_data(relation_data: List[Dict]) -> Dict:
    rels = dict()
    for d in relation_data:
        rels[d['relation']] = {
            'index': int(d['index']),
            'head_type': d['head_type'],
            'tail_type': d['tail_type'],
        }
    return rels

#convert head and tail entity、relation to dict index
def _add_relation_data(rels: Dict,data: List) -> None:
    for d in data:
        d['rel2idx'] = rels[d['relation']]['index']
        d['head_type'] = rels[d['relation']]['head_type']
        d['tail_type'] = rels[d['relation']]['tail_type']


def seq_len_to_mask(seq_len: Union[List, np.ndarray, torch.Tensor], max_len=None, mask_pos_to_true=True):
    
    if isinstance(seq_len, list):
        seq_len = np.array(seq_len)

    if isinstance(seq_len, np.ndarray):
        seq_len = torch.from_numpy(seq_len)

    if isinstance(seq_len, torch.Tensor):
        assert seq_len.dim() == 1, logger.error(f"seq_len can only have one dimension, got {seq_len.dim()} != 1.")
        batch_size = seq_len.size(0)
        max_len = int(max_len) if max_len else seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len.device)
        if mask_pos_to_true:
            mask = broad_cast_seq_len.ge(seq_len.unsqueeze(1))
        else:
            mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    else:
        raise logger.error("Only support 1-d list or 1-d numpy.ndarray or 1-d torch.Tensor.")

    return mask

##transform word to ids using bertTokenier 
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

# Preprocess,load data to pkl
logger.info('load raw files...')
train_fp = os.path.join('./semmedb_data//origin/train.csv')
valid_fp = os.path.join('./semmedb_data/origin/valid.csv')
test_fp = os.path.join('./semmedb_data/origin/test.csv')
relation_fp = os.path.join('./semmedb_data/origin/relation.csv')

train_data = load_csv(train_fp)
valid_data = load_csv(valid_fp)
test_data = load_csv(test_fp)
relation_data = load_relation_csv(relation_fp)
'''
for d in train_data:
    d['tokens'] = eval(d['tokens'])
for d in valid_data:
    d['tokens'] = eval(d['tokens'])
for d in test_data:
    d['tokens'] = eval(d['tokens'])
'''   
#handle relations data,convert head and tail entity and relation to index dict
logger.info('convert relation into index...')
rels = _handle_relation_data(relation_data)
_add_relation_data(rels, train_data)
_add_relation_data(rels, valid_data)
_add_relation_data(rels, test_data)

logger.info('verify whether use pretrained language models...')

#convert sentence to bert ids
logger.info('use pretrained language models serialize sentence...')
_lm_serialize(train_data, cfg)
_lm_serialize(valid_data, cfg)
_lm_serialize(test_data, cfg)

#saving to pkl
logger.info('save data for backup...')
os.makedirs(cfg.out_path, exist_ok=True)
train_save_fp = os.path.join(cfg.out_path, 'train.pkl')
valid_save_fp = os.path.join(cfg.out_path, 'valid.pkl')
test_save_fp = os.path.join(cfg.out_path, 'test.pkl')
save_pkl(train_data, train_save_fp)
save_pkl(valid_data, valid_save_fp)
save_pkl(test_data, test_save_fp)

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
        
        x, y = dict(), []
        word, word_len = [], []
        head_pos, tail_pos = [], []
        pcnn_mask = []
        adj_matrix = []
        for data in batch:
            word.append(_padding(data['token2idx'], max_len))
            word_len.append(int(data['lens']))
            y.append(int(data['rel2idx']))
            
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
        y = torch.tensor(y)
        
        if cfg.model_name != 'lm':
            x['head_pos'] = torch.tensor(head_pos)
            x['tail_pos'] = torch.tensor(tail_pos)
            if cfg.model_name == 'gcn':
                x['adj'] = torch.tensor(adj_matrix)
            if cfg.model_name == 'cnn' and cfg.use_pcnn:
                x['pcnn_mask'] = torch.tensor(pcnn_mask)

        return x, y
    
    return collate_fn_intra

#pytorch dataset
class CustomDataset(Dataset):
    def __init__(self, fp):
        self.file = load_pkl(fp)

    def __getitem__(self, item):
        sample = self.file[item]
        return sample

    def __len__(self):
        return len(self.file)

# pretrain language model
class PretrainLM(nn.Module):
    def __init__(self, cfg):
        super(PretrainLM, self).__init__()
        self.num_layers = cfg.rnn_layers
        self.lm = BertModel.from_pretrained(cfg.lm_file, num_hidden_layers=cfg.lm_num_hidden_layers)
        self.bilstm = nn.LSTM(768,10,batch_first=True,bidirectional=True,num_layers=cfg.rnn_layers,dropout=cfg.dropout)
        self.fc = nn.Linear(20, cfg.num_relations)
    # #forward-propagation
    def forward(self, x):
        N = self.num_layers
        word, lens = x['word'], x['lens']
        B = word.size(0)
        output, pooler_output = self.lm(word)
        output = pack_padded_sequence(output, lens, batch_first=True, enforce_sorted=True)
        _, (output,_) = self.bilstm(output)
        output = output.view(N, 2, B, 10).transpose(1, 2).contiguous().view(N, B, 20).transpose(0, 1)
        output = output[:,-1,:]
        output = self.fc(output)
        
        return output

#  p,r,f1 measurement metric 
class PRMetric():
    def __init__(self):
       
        self.y_true = np.empty(0)
        self.y_pred = np.empty(0)

    def reset(self):
        self.y_true = np.empty(0)
        self.y_pred = np.empty(0)

    def update(self, y_true:torch.Tensor, y_pred:torch.Tensor):
        y_true = y_true.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()
        y_pred = np.argmax(y_pred,axis=-1)

        self.y_true = np.append(self.y_true, y_true)
        self.y_pred = np.append(self.y_pred, y_pred)

    def compute(self):
        p, r, f1, _ = precision_recall_fscore_support(self.y_true,self.y_pred,average='macro',warn_for=tuple())
        _, _, acc, _ = precision_recall_fscore_support(self.y_true,self.y_pred,average='micro',warn_for=tuple())

        return acc,p,r,f1


# Iteration in training process
def train(epoch, model, dataloader, optimizer, criterion, cfg):
    model.train()
    model = model.cuda()

    metric = PRMetric()
    losses = []

    for batch_idx, (x, y) in enumerate(dataloader, 1):
        optimizer.zero_grad()
        
        x["word"],y = x["word"].cuda(),y.cuda()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        
        loss.backward()
        optimizer.step()

        metric.update(y_true=y, y_pred=y_pred)
        losses.append(loss.item())

        data_total = len(dataloader.dataset)
        data_cal = data_total if batch_idx == len(dataloader) else batch_idx * len(y)
        if (cfg.train_log and batch_idx % cfg.log_interval == 0) or batch_idx == len(dataloader):
            acc,p,r,f1 = metric.compute()
            print(f'Train Epoch {epoch}: [{data_cal}/{data_total} ({100. * data_cal / data_total:.0f}%)]\t'
                        f'Loss: {loss.item():.6f}')
            print(f'Train Epoch {epoch}: Acc: {100. * acc:.2f}%\t'
                        f'macro metrics: [p: {p:.4f}, r:{r:.4f}, f1:{f1:.4f}]')

    if cfg.show_plot and not cfg.only_comparison_plot:
        if cfg.plot_utils == 'matplot':
            plt.plot(losses)
            plt.title(f'epoch {epoch} train loss')
            plt.show()

    return losses[-1]


# Iteration in testing process
def validate(epoch, model, dataloader, criterion,verbose=True):
    model.eval()
    model = model.cuda()

    metric = PRMetric()
    losses = []

    for batch_idx, (x, y) in enumerate(dataloader, 1):
        with torch.no_grad():
            x["word"],y = x["word"].cuda(),y.cuda()
            y_pred = model(x)
            loss = criterion(y_pred, y)

            metric.update(y_true=y, y_pred=y_pred)
            losses.append(loss.item())

    loss = sum(losses) / len(losses)
    acc,p,r,f1 = metric.compute()
    data_total = len(dataloader.dataset)
    if verbose:
        print(f'Valid Epoch {epoch}: [{data_total}/{data_total}](100%)\t Loss: {loss:.6f}')
        print(f'Valid Epoch {epoch}: Acc: {100. * acc:.2f}%\tmacro metrics: [p: {p:.4f}, r:{r:.4f}, f1:{f1:.4f}]\n\n')

    return f1,loss

# Load dataset
train_dataset = CustomDataset(train_save_fp)
valid_dataset = CustomDataset(valid_save_fp)
test_dataset = CustomDataset(test_save_fp)

train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn(cfg))
valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn(cfg))
test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn(cfg))

# main entry, define optimization function, loss function and so on
# start epoch
# Use the loss of the valid dataset to make an early stop judgment. When it does not decline, this is the time when the model generalization is the best.
model = PretrainLM(cfg)
print(model)

#optimizer,scheduler,criterion
optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=cfg.lr_factor, patience=cfg.lr_patience)
criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()

#Initialization setting
best_f1, best_epoch = -1, 0
es_loss, es_f1, es_epoch, es_patience, best_es_epoch, best_es_f1, = 1000, -1, 0, 0, 0, -1
train_losses, valid_losses = [], []

#training
logger.info('=' * 10 + ' Start training ' + '=' * 10)
for epoch in range(1, cfg.epoch + 1):
    train_loss = train(epoch, model, train_dataloader, optimizer, criterion, cfg)
    #validate
    valid_f1, valid_loss = validate(epoch, model, valid_dataloader, criterion)
    #Adjusted learning rate
    scheduler.step(valid_loss)

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    #select best score
    if best_f1 < valid_f1:
        best_f1 = valid_f1
        best_epoch = epoch
    if es_loss > valid_loss:
        es_loss = valid_loss
        es_f1 = valid_f1
        best_es_f1 = valid_f1
        es_epoch = epoch
        best_es_epoch = epoch
        es_patience = 0
    else:
        es_patience += 1
        if es_patience >= cfg.early_stopping_patience:
            best_es_epoch = es_epoch
            best_es_f1 = es_f1
#plot
if cfg.show_plot:
    if cfg.plot_utils == 'matplot':
        plt.plot(train_losses, 'x-')
        plt.plot(valid_losses, '+-')
        plt.legend(['train', 'valid'])
        plt.title('train/valid comparison loss')
        plt.show()


print(f'best(valid loss quota) early stopping epoch: {best_es_epoch}, '
            f'this epoch macro f1: {best_es_f1:0.4f}')
print(f'total {cfg.epoch} epochs, best(valid macro f1) epoch: {best_epoch}, '
            f'this epoch macro f1: {best_f1:.4f}')

test_f1, _ = validate(0, model, test_dataloader, criterion,verbose=False)
print(f'after {cfg.epoch} epochs, final test data macro f1: {test_f1:.4f}')

#saving model
import os
print(os.getcwd())
with open("./data/out/biobert_model_%d_epoch.pkl"%cfg.epoch,"wb") as o:
    pickle.dump(model,o)

