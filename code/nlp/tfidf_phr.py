# -*- coding: utf-8 -*-
"""
this file calculated tfidf value for top popular phrases for each star(216 total)
return a 'tfidf_phrases.csv' file, each row refers to each review, each column refers to each phrases
"""

from textblob import TextBlob
import pandas as pd
import string
from nltk.corpus import stopwords
from collections import Counter
import itertools
import math
from tqdm import tqdm

hotel = pd.read_csv('data/hotel_train.csv') 
def text_process(text):
    '''
    Takes in a string of text, then performs the following:
    1. Detect the language of the text and translate it into english
    2. Remove all punctuation
    3. Remove all stopwords
    4. Return the cleaned text as a list of words
    '''
    s = unicode(text, "utf-8")
    #if detect(s)!='en':  #translate other language to english
        #blob = blob.translate(from_lang=blob.detect_language(),to='en')
     #   s = translator.translate(s).text
    nopunc = [char for char in s if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

def clean_text(text):
    text = " ".join(text)
    return(text)


hotel['words_clean'] = hotel['text'].apply(lambda x: text_process(x))
hotel['text_clean'] = hotel['words_clean'].apply(lambda x: clean_text(x))

def get_phrases(x):
    text = x.lower()
    blob = TextBlob(text)
    return blob.noun_phrases #set


hotel['phrases'] = hotel['text_clean'].apply(lambda x: get_phrases(x))
hotel['phrases'] = hotel['phrases'].apply(lambda x: list(x)) #contain all phrases
hotel['phrases_set'] = hotel['phrases'].apply(lambda x: list(set(x))) #the same phrase only appear once
hotel['phrases_count']=hotel['phrases'].apply(lambda x: Counter(x))# counter all phrases
hotel['len']=hotel['phrases'].apply(lambda x: float(len(x)))



###################idf
#create dataframe p contain all phrases and their idf value
p_list = list(itertools.chain(*list(hotel['phrases_set'])))
p_count = pd.DataFrame(Counter(p_list).most_common())
p = pd.DataFrame(p_count[p_count.columns[1]])
p.columns=['count']
p.index = p_count[p_count.columns[0]]

N=len(hotel)
def idf(d):
    return math.log(N/d)

p['idf'] = p['count'].apply(idf)
p.to_csv('idf_phrases.csv',encoding='utf-8')



#extract first 100 phrases for each star category
def count_phrase(stars):
    ph_list = list(itertools.chain(*list(hotel.loc[hotel['stars']==stars, 'phrases_set'])))
    ph_count = pd.DataFrame(Counter(ph_list).most_common())
    ph = pd.DataFrame(ph_count[ph_count.columns[1]])
    ph.index = ph_count[ph_count.columns[0]]
    ph.columns = [str(stars)]
    return ph
ph = pd.concat([count_phrase(1), count_phrase(2),count_phrase(3),count_phrase(4),count_phrase(5)],axis=1)


a=ph.sort_values(by=['1'], ascending=False).index[:100]
a=a.append(ph.sort_values(by=['2'], ascending=False).index[:100])
a=a.append(ph.sort_values(by=['3'], ascending=False).index[:100])
a=a.append(ph.sort_values(by=['3'], ascending=False).index[:100])
a=a.append(ph.sort_values(by=['5'], ascending=False).index[:100])
a=list(set(a))





############################TF

#create dataframe phrases contain term-frequency for particular phrases(a) in all review
phrases_to_keep=a #input phrases
phrases = pd.DataFrame()
for phr in tqdm(phrases_to_keep):
        phrases[phr] = [sublist[phr] if phr in sublist else 0
                                    for sublist in hotel['phrases_count']]

for col in phrases.columns:
    phrases[col] = phrases[col]/hotel['len']
phrases.to_csv('tf_phrases.csv',encoding='utf-8')

############################TFIDF
tfidf=phrases.copy()
for col in tfidf.columns:
    tfidf[col]=tfidf[col]*p.loc[col,'idf']
tfidf.to_csv('tfidf_phrases.csv',encoding='utf-8')
