#encoding: utf-8
import codecs
import pandas as pd
import numpy as np
import re
import pickle   
import os
import collections
import jieba 
from pytorch_pretrained_bert import BertTokenizer

SENTENCE_LENGTH = 70

def data2pkl(filein,bert_token=False):
	datas = list()
	labels = list()
	lens = list()
	linedata=list()
	linelabel=list()
	word_list = []
	tags = ['O','B-A','I-A','B-O','I-O']

	# charembedding = []
	# with open('../data/charembeddings.pkl','rb') as f:
	#     chars = pickle.load(f)
	#     embeddings = pickle.load(f)   

	input_data = codecs.open(filein,'r','utf-8')
	for line in input_data.readlines():
	    line = line.split()
	    linedata=[]
	    linelabel=[]
	    numNotO=0
	    for word in line:
	        word = word.split('/')
	        if word[0]!='' and word[1]!='':    #这里句子中含有字符 / ,所以在以/作为分隔符时出现了问题
		        linedata.append(word[0])
		        linelabel.append(word[1])
	        # assert word[1] in tags,'****There is a tag is not in predefined tags list!{}--'.format(word[0])
	        if word[1]=='':
	            numNotO+=1
	    # if numNotO!=0:
	    lens.append(len(linedata))
	    datas.append(linedata)
	    labels.append(linelabel)
	    word_list += linedata
	    # if numNotO>0:
	    # 	print(numNotO)
	    # 	print(line)
	    # 	assert 1<0
	input_data.close()
	
	sr_allwords = pd.Series(word_list)
	sr_allwords = sr_allwords.value_counts()		#相当于给所有单词按照词频排序
	set_words = sr_allwords.index
	top2 = ['<PAD>','<UNK>']
	vocab = top2+set_words.tolist()
	set_ids = range(1, len(set_words)+1)
	# with open('../data/split/vocabulary.txt','w+',encoding='utf-8') as f:
	# 	# f.writelines(vocab)
	# 	for v in vocab:
	# 		f.write(v+'\n')
    #生成外部词向量的embeddings
    # charembedding.append([0.]*300)
    # for w in set_words:
    #     if w in chars:
    #         charembedding.append(embeddings[chars.index(w)])
    #     else:
    #         print('this char is not included in pretrained embeddings!')
    #         charembedding.append(list(np.random.rand(300)))
    # charembedding = np.array(charembedding)
    # with open('../data/fitembedding.pkl','wb') as f:
    #     pickle.dump(charembedding,f)

	tags = [i for i in tags]
	tag_ids = range(len(tags))
	word2id = pd.Series(set_ids, index=set_words)
	id2word = pd.Series(set_words, index=set_ids)
	tag2id = pd.Series(tag_ids, index=tags)
	id2tag = pd.Series(tags, index=tag_ids)
	# print(tag2id)
	# print( word2id)
	max_len = SENTENCE_LENGTH
	def X_padding(words):
		ids = []
		allwords = set(word2id.index)
		for w in words:
			if w in allwords:
				ids.append(word2id[w])
			else:
				ids.append(0)
		if len(ids) >= max_len:
		    return ids[:max_len]
		ids.extend([0]*(max_len-len(ids))) 
		return ids

	def X_bert(w):
		tokens = tokenizer.tokenize(w) #if w not in ("[CLS]", "[SEP]") else [w]
		ids = tokenizer.convert_tokens_to_ids(tokens)
		if len(ids) >= max_len:
		    return ids[:max_len]
		ids.extend([0]*(max_len-len(ids))) 
		return ids

	def y_padding(tags):
	    ids = list(tag2id[tags])
	    if len(ids) >= max_len: 
	        return ids[:max_len]
	    ids.extend([0]*(max_len-len(ids))) 
	    return ids
	df_data = pd.DataFrame({'words': datas, 'tags': labels}, index=range(len(datas)))
	if bert_token:
		tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=False)
		df_data['x'] = df_data['words'].apply(X_bert)
	else:
		df_data['x'] = df_data['words'].apply(X_padding)
	df_data['y'] = df_data['tags'].apply(y_padding)
	x = np.array(list(df_data['x'].values))
	y = np.array(list(df_data['y'].values))

	with open('../data/split/train2data.pkl', 'wb') as outp:
	    pickle.dump(word2id, outp)
	    pickle.dump(id2word, outp)
	    pickle.dump(tag2id, outp)
	    pickle.dump(id2tag, outp)
	    pickle.dump(x, outp)
	    pickle.dump(y, outp)
	    pickle.dump(lens,outp)

	print('***All string data have been transformed to numerical data in train2data.pkl!')



def trans2id(filein,bert_token=False):
	datas = list()
	labels = list()
	lens = list()
	linedata=list()
	linelabel=list()
	tags = ['O','B-A','I-A','B-O','I-O']

	input_data = codecs.open(filein,'r','utf-8')
	for line in input_data.readlines():
	    line = line.split()
	    linedata=[]
	    linelabel=[]
	    numNotO=0
	    for word in line:
	        word = word.split('/')
	        if word[0]!='' and word[1]!='':    #这里句子中含有字符 / ,所以在以/作为分隔符时出现了问题
		        linedata.append(word[0])
		        linelabel.append(word[1])
	        # assert word[1] in tags,'****There is a tag is not in predefined tags list!{}--'.format(word[0])
	        if word[1]=='':
	            numNotO+=1
	    lens.append(len(linedata))
	    datas.append(linedata)
	    labels.append(linelabel)
	input_data.close()

	with open('../data/split/train2data.pkl', 'rb') as p:
	    word2id = pickle.load(p)
	    id2word = pickle.load(p)
	    tag2id = pickle.load(p)

	max_len = SENTENCE_LENGTH
	def X_padding(words):
		ids = []
		allwords = set(word2id.index)
		for w in words:
			if w in allwords:
				ids.append(word2id[w])
			else:
				ids.append(0)
		if len(ids) >= max_len:  
		    return ids[:max_len]
		ids.extend([0]*(max_len-len(ids))) 
		return ids

	def X_bert(w):
		tokens = tokenizer.tokenize(w) #if w not in ("[CLS]", "[SEP]") else [w]
		ids = tokenizer.convert_tokens_to_ids(tokens)
		if len(ids) >= max_len:
		    return ids[:max_len]
		ids.extend([0]*(max_len-len(ids))) 
		return ids

	def y_padding(tags):
	    ids = list(tag2id[tags])
	    if len(ids) >= max_len: 
	        return ids[:max_len]
	    ids.extend([0]*(max_len-len(ids))) 
	    return ids
	df_data = pd.DataFrame({'words': datas, 'tags': labels}, index=range(len(datas)))
	if bert_token:
		tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=False)
		df_data['x'] = df_data['words'].apply(X_bert)
	else:
		df_data['x'] = df_data['words'].apply(X_padding)
	df_data['y'] = df_data['tags'].apply(y_padding)
	x = np.array(list(df_data['x'].values))
	y = np.array(list(df_data['y'].values))

	with open('../data/split/test2data.pkl','wb') as f:
		pickle.dump(x,f)
		pickle.dump(y,f)
		pickle.dump(lens,f)
	print('***all test data has been transformed')
	return x,y,lens


def partition(x,y,epochs,batch_size,lens=None):
    data_size = len(y)
    batch_num = int(data_size/batch_size) + 1
    for i in range(epochs):
        shuffled_indices = np.random.permutation(data_size)
        x = x[shuffled_indices]
        y = y[shuffled_indices]
        if lens is not None:
        	lens = lens[shuffled_indices]
        start = 0
        for j in range(batch_num):
            start = j*batch_size
            end = min(start+batch_size,data_size)
            if lens is None:
            	yield x[start:end],y[start:end]
            else:
            	yield x[start:end],y[start:end],lens[start:end]


if __name__ == "__main__":
	data2pkl('../data/split/train.csv',bert_token=False)
	trans2id('../data/split/dev.csv',bert_token=False)