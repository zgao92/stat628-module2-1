#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 15:27:27 2019

@author: xiajian
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer

train = pd.read_csv('data/train.csv',nrows=10000) #this take a while
train.head(5)
list(train.columns.values)

#check for duplicate data by text column,there is only two duplicate samples, 7690 and 9697, check their other value, only date are different,so we delete one of them.

dup=train[train.duplicated('text',keep=False)].sort_values('text')
train = train.drop_duplicates('text')

#natural language process 1
#use the English stop words list that is defaulted in the sklearn library
#add punc to this predefined list
punc = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}',"%",'_']
stop_words = text.ENGLISH_STOP_WORDS.union(punc)
desc = train['text'].values
y=train['stars']

#make text as vector
#different form of word show up, digits show up
vectorizer = TfidfVectorizer(stop_words = stop_words)
X = vectorizer.fit_transform(desc)

word_features = vectorizer.get_feature_names()
word_features[:100]
word_features[1000:1100]

#remove digit
word_features=[x for x in word_features if not any(c.isdigit() for c in x)]
word_features[:100]

# only the word root form is present
#SnowballStemmer() reduce words to root forms
#RegexpTokenizer() Regular expressions are a powerful tool in finding patterns in strings, which helps us find the tokens we want
stemmer = SnowballStemmer('english')
tokenizer = RegexpTokenizer(r'[a-zA-Z\']+')

def tokenize(text):
    return [stemmer.stem(word) for word in tokenizer.tokenize(text.lower())]

#limited our token feature space with regular expressions,decrease the dimension of vector
#selects only the top max_features tokens ordered by their frequencies in the corpus to be included in the vectorizing.

vectorizer2 = TfidfVectorizer(stop_words = stop_words, tokenizer = tokenize,max_features = 1000)

#got vectorized text X2
X2 = vectorizer2.fit_transform(desc)

#output features
word_features2 = vectorizer2.get_feature_names()




#natural language process 2
#try spacy package
import spacy
