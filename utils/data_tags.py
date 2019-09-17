#encoding: utf-8
import pandas as pd
import numpy as np
import jieba
import os

#file in:
TRAIN_REVIEW_FILE = '../data/split/train_reviews.csv'
TRAIN_LABEL_FILE  = '../data/split/train_labels.csv'
DEV_REVIEW_FILE = '../data/split/dev_reviews.csv'
DEV_LABEL_FILE = '../data/split/dev_labels.csv'
#file out:
TRAIN_FILE = '../data/split/train.csv'
#	TRAIN_FILE中的tag为：   B\I-A\O
DEV_FILE   = '../data/split/dev.csv'

def trans2pipeline(review_file,label_file,out_file):
	train_reviews = pd.read_csv(review_file).dropna()
	train_labels  = pd.read_csv(label_file).dropna()
	for i in range(len(train_reviews)):
		review_id = train_reviews.iloc[i]['id']
		# print(review_id)
		# assert 1<0
		text = train_reviews.iloc[i]['Reviews']
		tags = ['O'] * len(text)
		sub_labels = train_labels[train_labels['id']==review_id]
		for j in range(len(sub_labels)):
			if sub_labels.iloc[j]['AspectTerms'] == '_':
				pass
			else:
				A_start = int(sub_labels.iloc[j]['A_start'])
				A_end   = int(sub_labels.iloc[j]['A_end'])
				tags[A_start] = 'B-A'
				tags[A_start+1:A_end] = ['I-A'] * (A_end - A_start - 1)

			if sub_labels.iloc[j]['OpinionTerms'] == '_':
				# print('{}\t\t{}\t\t{}\n'.format(review_id,text,polarity))
				pass
			else:
				O_start = int(sub_labels.iloc[j]['O_start'])
				O_end   = int(sub_labels.iloc[j]['O_end'])
				tags[O_start] = 'B-O'
				tags[O_start+1:O_end] = ['I-O'] * (O_end - O_start - 1)

		with open(out_file,'a+',encoding='utf-8') as f:
			# f.write('#id = {}\n'.format(review_id))
			for k in range(len(text)):
				f.write('{}/{} '.format(text[k],tags[k]))
			f.write('\n')

trans2pipeline(TRAIN_REVIEW_FILE,TRAIN_LABEL_FILE,TRAIN_FILE)
trans2pipeline(DEV_REVIEW_FILE,DEV_LABEL_FILE,DEV_FILE)