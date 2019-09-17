# coding=utf-8
import pickle
import pdb
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import codecs 
import numpy as np 
import os
from pytorch_pretrained_bert import BertTokenizer
import argparse

from utils.data_process import trans2id,partition
from models.biLSTM_CRF import BiLSTMCRF
from models.bert_token import BertNet
from data.result.resultCal import calculate2

#-------------------基本设置---------------------
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
np.random.seed(2019)
START_TAG = "<START>"
STOP_TAG = "<STOP>"

#-------------------数据下载---------------------
with open('./data/split/train2data.pkl', 'rb') as inp:
    word2id = pickle.load(inp)
    id2word = pickle.load(inp)
    tag2id = pickle.load(inp)
    id2tag = pickle.load(inp)
    x_train = torch.from_numpy(pickle.load(inp))
    y_train = torch.from_numpy(pickle.load(inp))
    len_train = torch.tensor(pickle.load(inp))

with open('./data/split/test2data.pkl','rb') as f:
    x_test = torch.from_numpy(pickle.load(f))
    y_test = torch.from_numpy(pickle.load(f))
    len_test = torch.tensor(pickle.load(f))

# with open('./data/split/bertembedding.pkl', 'rb') as f:
#     weights = pickle.load(f)
print ("Train len:",len(x_train))
print('Test len:',len(x_test))
print('Vocab size:',len(word2id))   #词典里不包含UNK
print('***Finsh data loading!\n')

vocab_size,tag_size = len(id2word),len(tag2id)

def test(sentence,tags,lens,model,epoch,use_bert):
    entityres=[]
    entityall=[]
    sentence = sentence.long()
    if use_bert:
        _,predict = model(sentence,training=False)
        predicts = predict.cpu().detach().numpy().tolist()
        ids2words = tokenizer
    else:
        predicts = model.predict(sentence,lens)
        ids2words = id2word
    tags = tags.cpu().detach().numpy().tolist()
    for senten,pre,tag in zip(sentence,predicts,tags):
        entityres = calculate2(senten,pre,ids2words,id2tag,entityres,bert_token=use_bert)
        entityall = calculate2(senten,tag,ids2words,id2tag,entityall,bert_token=use_bert)
    jiaoji = [i for i in entityres if i in entityall]
    print('=========This is the {} epoch=========='.format(epoch))
    if len(jiaoji)!=0:
        precision =  float(len(jiaoji))/len(entityres)
        recall = float(len(jiaoji))/len(entityall)
        f = 2*precision*recall/(precision+recall)
        print ("precision:", precision)
        print ("recall:", recall)
        print ("f:", (2*precision*recall)/(precision+recall))
    else:
        print ("precision:",0)
        f = 0
    print('\n')
    return f


# L1_loss = torch.tensor(0.).cuda()
# L2_loss = torch.tensor(0.).cuda()
# for params in model.parameters():
#     L1_loss += Variable(torch.norm(params,1))
#     L2_loss += Variable(torch.norm(params,2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',type=str,default='BiLSTMCRF')
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--lr',type=float,default=0.0001)
    parser.add_argument('--embedding_dim',type=int,default=300)
    parser.add_argument('--hidden_dim',type=int,default=200)
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--lstm_layers',type=int,default=1)
    parser.add_argument('--lambda1',type=float,default=0.5)
    parser.add_argument('--lambda2',type=float,default=0.01)

    config = parser.parse_args()
    DEVICE = torch.device('cuda',config.gpu)
    model_name = config.model

    if model_name == 'BiLSTMCRF':
        use_bert = False
        model = BiLSTMCRF(tag_size,vocab_size,config.embedding_dim,config.hidden_dim,config.lstm_layers).to(DEVICE)
        
    elif model_name == 'BertNet':
        use_bert = True
        model = BertNet(TAG_SIZE,top_rnns=False,device=DEVICE,finetuning=True).to(DEVICE)
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        
    else:
        print('*Error:The model is not existed!')
        exit(1)
    print('=========The model {} is initialized successfully!=========='.format(model_name))

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-4)
    trainbatch = partition(x_train,y_train,config.epochs,config.batch_size,len_train)

    iters = 0
    best_f = 0.
    for sentence, tags,lens in trainbatch:
        sentence = sentence.to(DEVICE)
        tags = tags.to(DEVICE)
        lens = lens.to(DEVICE).long()
        model.zero_grad()
        if not use_bert:
            loss = model(sentence,lens, tags)
        else:
        #BertNet
            cross_loss = nn.CrossEntropyLoss()
            logits,_ = model(sentence)
            logits = logits.view(-1,logits.shape[2])
            tags = tags.view(-1)
            loss = cross_loss(logits,tags) #+ LABMDA1 * L1_loss + LABMDA2 * L2_loss
        loss.backward()
        optimizer.step()
        iters += 1
        if iters%config.batch_size == 0:
            f = test(x_test.to(DEVICE),y_test.to(DEVICE),len_test.to(DEVICE),model,iters//config.batch_size,use_bert)
            if f>best_f: 
                best_f = f
                torch.save(model,'model.pkl/{}.pkl'.format(model_name))
    print('The highest f score is {}'.format(best_f))
    os.rename('model.pkl/{}.pkl'.format(model_name),'model.pkl/{}_{}.pkl'.format(model_name,best_f))
    print('***The best model has been saved!')

# fine-tuning
# model = torch.load('./model0.8732286480275756.pkl')
# nanum = 0
# for _ in range(5):
#     for sentence, tags in zip(x_test,y_test):
#         model.zero_grad()

#         sentence=torch.tensor(sentence, dtype=torch.long).cuda()
#         # tags=torch.tensor(tags, dtype=torch.long).cuda()
#         try:
#             tags = torch.tensor([tag2id[int(t)] for t in tags], dtype=torch.long).cuda()
#         except:
#             nanum += 1
#             continue
#         loss = model.neg_log_likelihood(sentence, tags)

#         loss.backward()
#         optimizer.step()
#     for sentence, tags in zip(x_valid,y_valid):
#         model.zero_grad()

#         sentence=torch.tensor(sentence, dtype=torch.long).cuda()
#         # tags=torch.tensor(tags, dtype=torch.long).cuda()
#         try:
#             tags = torch.tensor([tag2id[int(t)] for t in tags], dtype=torch.long).cuda()
#         except:
#             nanum += 1
#             continue
#         loss = model.neg_log_likelihood(sentence, tags)

#         loss.backward()
#         optimizer.step()

# torch.save(model,'./piped_models/extractmodel.pkl')