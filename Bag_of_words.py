# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 17:52:54 2020

@author: Harshil
"""

#Importing Libraries
import numpy as np
import pandas as pd
import re

#Importing Dataset
dataset=pd.read_csv("labelled_data.tsv", delimiter="\t", quoting=3)
dataset['class'].value_counts()
X=dataset.iloc[:,1].values
y=dataset.iloc[:,0].values

#imbalanced data
count_values=pd.value_counts(y, sort=True)
count_values.plot(kind='bar', rot=0)

#Random Under Sampling
X=X.reshape(-1,1)
import imblearn
from imblearn.under_sampling import RandomUnderSampler
rus=RandomUnderSampler(sampling_strategy="auto")
X_res, y_res=rus.fit_resample(X, y)

#Random Over Sampling
"""X=X.reshape(-1,1)
import imblearn
from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler(sampling_strategy="auto")
X_ros, y_ros=ros.fit_sample(X, y)"""

#Balanced data
count_values=pd.value_counts(y_res, sort=True)
count_values.plot(kind='bar', rot=0)

#Cleaning the text
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0, 7906):
    tweet=X_res[i][0]
    tweet=' '.join(word for word in tweet.split(' ') if not word.startswith('http'))
    tweet=re.sub('[^a-zA-Z@#&]', ' ' , tweet)
    tweet=' '.join(word for word in tweet.split(' ') if not word.startswith('@'))
    tweet=' '.join(word for word in tweet.split(' ') if not word.startswith('&'))
    tweet=' '.join(word for word in tweet.split(' ') if not word.startswith('#'))
    tweet=tweet.lower()
    tweet=tweet.split()
    
    ps=PorterStemmer()
    tweet=[ps.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
    tweet=" ".join(tweet)
    corpus.append(tweet)
       
#Creating bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X=cv.fit_transform(corpus).toarray()
y=[y_res[i] for i in range(0, 7906)]
