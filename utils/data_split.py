#encoding: utf-8
import pandas as pd
import numpy as np
import pickle

np.random.seed(2019)

reviews = pd.read_csv('../data/origin/Train_reviews.csv')
labels = pd.read_csv('../data/origin/Train_labels.csv')

DATA_SIZE = len(reviews)
TRAIN_SIZE = int(DATA_SIZE*0.7)
DEV_SIZE = DATA_SIZE - TRAIN_SIZE

shuffled_dices = np.random.permutation(DATA_SIZE)
reviews = reviews.iloc[shuffled_dices]
train_reviews = reviews.iloc[:TRAIN_SIZE]
train_labels = labels[labels['id'].isin(train_reviews['id'])]
dev_reviews = reviews.iloc[TRAIN_SIZE:]
dev_labels = labels[labels['id'].isin(dev_reviews['id'])]

# print(len(train_reviews))  	#2260
# print(len(train_labels))	#4661
# print(len(dev_reviews))		#969
# print(len(dev_labels))		#1972

train_reviews.to_csv('../data/split/train_reviews.csv',index=None)
train_labels.to_csv('../data/split/train_labels.csv',index=None)
dev_reviews.to_csv('../data/split/dev_reviews.csv',index=None)
dev_labels.to_csv('../data/split/dev_labels.csv',index=None)

#由于打标记的文件中不含有id信息，所以这里记一下
with open('../data/split/reviewid.pkl','wb') as f:
	pickle.dump(train_reviews['id'].tolist(),f)
	pickle.dump(dev_reviews['id'].tolist(),f)
# print(dev_reviews['id'].tolist())
#为了之后计算方便，将乱序的id映射为有序的
# oldid2newid = pd.Series(range(1,TRAIN_SIZE+1),index=train_reviews['id'])
# # for i in range(len(train_labels)):
# # 	oldid = train_labels.iloc[i]['id']
# # 	train_labels.iloc[i]['id'] = oldid2newid[oldid]
# print(train_labels.iloc[7])
# (train_labels.iloc[7])['id'] = 123
# print(train_labels.at[7,'id'])
# print(train_labels.iloc[7])
