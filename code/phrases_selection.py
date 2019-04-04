"""This file is the summary of all the code in the jupyter notebook.

To:
    1. get few plots for data overview
    2. translate foreign language into english
    3. extract phrases for each reviews
    4. get important negative/positive phrases by spearman correlation
    5. output importance_phrases_hotel.csv file which include the frequency
       rate of phrases under
       each stars level and their spearman correlation coefficient.

@author: Ke Tang
"""

import ast
import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
#import plotly
#plotly.offline.init_notebook_mode(connected=True)
#import plotly.offline as py
import plotly.plotly as py
import plotly.tools as tls
import string
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from src.data import read_files
from collections import Counter
import itertools
from textblob import TextBlob
from langdetect import detect

import string
from googletrans import Translator
#from google.api_core.protobuf_helpers import get_messages
#from __future__ import absolute_import
from google.cloud import translate
from langdetect import detect
import os
from scipy.stats import spearmanr

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import csv

######functions#######

def text_process(text):
    '''
    Takes in a string of text, then performs the following:
    1. Detect the language of the text and translate it into english
    2. Remove all punctuation
    3. Remove all stopwords
    4. Return the cleaned text as a list of words
    '''
    s = unicode(text, "utf-8")
    s = TextBlob(s)
    s = s.correct()
    try:
        if detect(s)!='en':  #translate other language to english
            s = translate_client.translate(s, target_language='en')['translatedText']
    except:
        s = s
    nopunc = [char for char in s if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

def get_phrases(x):
    text = x.lower()
    blob = TextBlob(text)
    return set(blob.noun_phrases)

def count_fre(stars, ch, df):
    """
    count the frequency of 'ch': could be phrases or words
    corresponding to the 'stars': could be 1, 2, 3, 4, 5
    """
    p_list = list(itertools.chain(*list(df.loc[df['stars']==stars, ch])))
    p_count = pd.DataFrame(Counter(p_list).most_common())
    p = pd.DataFrame(p_count[p_count.columns[1]])
    p.index = p_count[p_count.columns[0]]
    p.columns = [str(stars)]
    return p

def str_to_list(s):
    """
    transform a string to a list, e.g. "[a,b,c]" to [a,b,c]
    """
    s = s.replace("u'", "")
    s = s.replace("'", "")
    l = s[1:-1].split(', ')
    return l

def clean_word_set(x):
    """
    to delete ' , [ ] that showed after setting the unique words
    """
    s = [i.replace("'",'') for i in x]
    s = [i.replace(",",'') for i in s]
    s = [i.replace("[",'') for i in s]
    s = [i.replace("]",'') for i in s]
    return s

def get_spearman_corr(row):
    """
    calculate the spearman's correlation coeffecient: for both linear and nonlinear relation
    """
    x = [row['1'],row['2'],row['3'],row['4'],row['5']]
    return spearmanr(x,range(1,6))[0]
##########################################


#######execute code#####
train = pd.read_csv('data/train.csv', index_col=0)
train['category'] = 'Other'
train.loc[train.iloc[:,11]==1, 'category'] = 'Restaurants'
train.loc[train.iloc[:,18]==1, 'category'] = 'EventPlanning&Services'
train.loc[train.iloc[:,19]==1, 'category'] = 'Shopping'
train.loc[train.iloc[:,21]==1, 'category'] = 'Beauty&Spas'
train.loc[train.iloc[:,27]==1, 'category'] = 'Hotels&Travel'
train.loc[train.iloc[:,31]==1, 'category'] = 'HomeServices'
train.loc[train.iloc[:,33]==1, 'category'] = 'Automotive'
train.drop(train.index[3170346],inplace=True)

train['stars'] = train['stars'].astype(float).astype(int)
ax = sns.barplot(x="stars", y="category", data=train, order=['Beauty&Spas','EventPlanning&Services',
                                                             'Other','Restaurants','Shopping','HomeServices',
                                                            'Automotive','Hotels&Travel'])
plt.title('Average stars vs. categories')
plt.show()

#text cleaning
#23 languages besides english, with 1006 reviews in foreign language
hotel = pd.DataFrame.from_csv('./data/hotel_train.csv')
path = 'data/translation.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = path
translate_client = translate.Client()
hotel['words_clean'] = hotel['text'].apply(lambda x: text_process(x))
hotel['text_clean'] = hotel['words_clean'].apply(lambda x: ' '.join(x))
hotel['phrases'] = hotel['text_clean'].apply(lambda x: get_phrases(x))
hotel['words_clean'] = hotel['words_clean'].apply(lambda x: set(str(x).split(' ')))     #get unique word shown in each review

p = pd.concat([count_fre(1, 'phrases', hotel), count_fre(2,'phrases', hotel),count_fre(3,'phrases', hotel),count_fre(4,'phrases', hotel),
               count_fre(5,'phrases', hotel)],axis=1)
p = p.sort_values(by=['1'], ascending=False)
p = p.fillna(0)

#w = pd.concat([count_fre(1, 'words_clean', hotel), count_fre(2,'words_clean', hotel),count_fre(3,'words_clean', hotel),count_fre(4,'words_clean', hotel),
#               count_fre(5,'words_clean', hotel)],axis=1)
#w = w.sort_values(by=['1'], ascending=False)
#w = w.fillna(0)

#scale the importance of features under each stars level
p.ix[:,'1':'5'] = p.ix[:,'1':'5']/Counter(hotel['stars']).values()
#w.ix[:,'1':'5'] = w.ix[:,'1':'5']/Counter(hotel['stars']).values()

p['correlation'] = p.apply(get_spearman_corr,axis=1)
#w['correlation'] = w.apply(get_spearman_corr,axis=1)

p.to_csv('./data/importance_phrases_hotel.csv',index=True)
