#encoding: utf-8
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import pickle

from pytorch_pretrained_bert import BertTokenizer,BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=False)
bert = BertModel.from_pretrained('bert-base-chinese')

embeddings = bert.embeddings.word_embeddings.weight
print(type(embeddings))

print(embeddings.shape)

vocab = []
with open('../data/split/vocabulary.txt','r',encoding='utf-8') as f:
	f.readline()
	f.readline()
	lines = f.readlines()
	for line in lines:
		vocab.append(line.strip())
print('the vocabulary size is',len(vocab))

notinbert = []
weights = [np.zeros(768)]
for word in vocab:
	try:
		token = tokenizer.tokenize(word)
		ids = tokenizer.convert_tokens_to_ids(token)
		word_embedding = embeddings[ids[0]]
		weights.append(word_embedding.detach().numpy())
	except:
		notinbert.append(word)
		print(word)
		weights.append(np.zeros(768))
	
print('words not in the vocabulary:',notinbert)
print(len(weights))
with open('../data/split/bertembedding.pkl','wb') as f:
	pickle.dump(np.array(weights),f)
