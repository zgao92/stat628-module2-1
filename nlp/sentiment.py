#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 21:54:31 2019

@author: xiajian
"""

import pandas as pd
from nltk.stem.snowball import SnowballStemmer
import re
from nltk.corpus import stopwords

train = pd.read_csv('data/train.csv',nrows=10000) #try a small amount of sample
desc = train['text'].values
y=train['stars']

#split text to list of words,including Removing alphanumeric, spelling and punctuation characters, lowercase, remove stop words, and stem words
def split_text(text):
    """split text to list of words

    Args:
        text: srt, an article

    Returns:
        com: a list of words.
    """
    com = re.sub('[^a-zA-Z]', ' ', text)
    com = com.lower()
    com = com.split()
    stemmer = SnowballStemmer('english')
    com = [stemmer.stem(word) for word in com if not word in set(stopwords.words('english'))]
    com = list(set(com))
    return com



def count_word(desc,y):
    """count the occurance of each word in each class

    Args:
        desc: an array of text
        y: an array of stars

    Returns:
        BOW_df:a DataFrame containing the occurance of each word in each class
    """
    words_set = set()
    BOW_df = pd.DataFrame(columns=['1.0', '2.0', '3.0','4.0','5.0'])

    for i in range(len(desc)):
        com=split_text(desc[i])
        stars=str(y[i])
        for word in com:
            if word not in words_set:
                words_set.add(word)
                BOW_df.loc[word] = [0,0,0,0,0]
                BOW_df.ix[word][stars] += 1
            else:
                BOW_df.ix[word][stars] += 1
    return BOW_df