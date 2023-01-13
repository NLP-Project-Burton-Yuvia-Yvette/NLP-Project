import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nltk
import unicodedata
import re
import requests


import nltk.sentiment

from bs4 import BeautifulSoup
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Optional, Union, cast
import requests
from env import github_token, github_username
import prepare as p
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from scipy import stats
import acquire



def get_sentiment(sentiment_df):
    
    # reindex dataframe
    sentiment_df.reset_index(drop =True, inplace=True)
    #create sntiment object
    sia = nltk.sentiment.SentimentIntensityAnalyzer()
    
    # create row id column
    sentiment_df["row_id"] =sentiment_df.index +1
    
    # create subsets
    df_subset = sentiment_df[['row_id', 'clean_text']].copy()
    
    # set up empty dataframe for staging output
    df1=pd.DataFrame()
    df1['row_id']=['99999999999']
    df1['sentiment_type']='NA999NA'
    df1['sentiment_score']=0
    
    # run loop to calculate and save sentiment values
    t_df = df1
    for index,row in df_subset.iterrows():
        scores = sia.polarity_scores(row[1])
        for key, value in scores.items():
            temp = [key,value,row[0]]
            df1['row_id']=row[0]
            df1['sentiment_type']=key
            df1['sentiment_score']=value
            t_df=t_df.append(df1)
    #remove dummy row with row_id = 99999999999
    t_df_cleaned = t_df[t_df.row_id != '99999999999']
    #remove duplicates if any exist
    t_df_cleaned = t_df_cleaned.drop_duplicates()
    # only keep rows where sentiment_type = compound
    t_df_cleaned = t_df[t_df.sentiment_type == 'compound']
    
    df_output = pd.merge(sentiment_df, t_df_cleaned, on='row_id', how='inner')
    
    
    plt.subplot(1, 2, 1)
    sns.kdeplot(df_output[df_output.language == 'JavaScript'].sentiment_score, label = 'JavaScript')
    sns.kdeplot(df_output[df_output.language == "Python"].sentiment_score, label = 'Python')
    sns.kdeplot(df_output[df_output.language == "Java"].sentiment_score, label = 'Java')
    sns.kdeplot(df_output[df_output.language == "TypeScript"].sentiment_score, label = 'TypeScript')

    plt.legend(['JavaScript', 'Python','Java','TypeScript'])
    
    
    #generate mean of sentiment_score by period
    plt.subplot(1, 2, 2)
    dfg = df_output.groupby(['language'])['sentiment_score'].mean()
    #create a bar plot
    dfg.plot(kind='bar', title='Sentiment Score', ylabel='Mean Sentiment Score',
         xlabel='Period', figsize=(6, 5))
   
    
    return plt.show();
    
def number_words(train):
    JavaScript_words = ' '.join(train[train.language == 'JavaScript'].clean_text)
    Java_words = ' '.join(train[train.language == 'Java'].clean_text)
    Python_words = ' '.join(train[train.language == 'Python'].clean_text)
    TypeScript_words = ' '.join(train[train.language == 'TypeScript'].clean_text)
    all_words = ' '.join(train.clean_text)
    return JavaScript_words, Java_words, Python_words, TypeScript_words, all_words

def frequency_of_words(JavaScript_words, Java_words, Python_words, TypeScript_words, all_words):
    JavaScript_freq1 = pd.Series(JavaScript_words).value_counts()
    Java_freq1 = pd.Series(Java_words).value_counts()
    Python_freq1 = pd.Series(Python_words).value_counts()
    TypeScript_freq1 = pd.Series(TypeScript_words).value_counts()
    all_freq1 = pd.Series(all_words).value_counts()
    JavaScript_freq = pd.Series(str(JavaScript_freq1).split(' '))
    Java_freq = pd.Series(str(Java_freq1).split(' '))
    Python_freq = pd.Series(str(Python_freq1).split(' '))
    TypeScript_freq = pd.Series(str(TypeScript_freq1).split(' '))
    all_freq = pd.Series(str(all_freq1).split(' '))
    return JavaScript_freq, Java_freq, Python_freq, TypeScript_freq, all_freq

    
def bar_common_language(train):
    fig = plt.figure(figsize = (10, 5))
    ax = fig.add_axes([0,0,1,1])
    langs = ['Java', 'JavaScript', 'Python', 'TypeScript']
    language = [len(train[train.language == 'Java']), len(train[train.language == 'JavaScript']), len(train[train.language == 'Python']), len(train[train.language == 'TypeScript'])]
    ax.bar(langs,language, color = 'blue')
    plt.xlabel("Language used")
    plt.ylabel(" ")
    plt.title("Language most commonly used")

    plt.show()
    
    
def bar_average_word(train, JavaScript_freq, Java_freq, Python_freq, TypeScript_freq, all_freq):
    fig = plt.figure(figsize = (10, 5))
    ax = fig.add_axes([0,0,1,1])
    langs = ['All','Java', 'JavaScript', 'Python', 'TypeScript']
    language = [all_freq.count()/len(train.language), Java_freq.count()/len(train[train.language == 'Java']), JavaScript_freq.count()/len(train[train.language == 'JavaScript']), Python_freq.count()/len(train[train.language == 'Python']), TypeScript_freq.count()/len(train[train.language == 'TypeScript'])]
    ax.bar(langs,language, color = 'blue')
    plt.xlabel("Language used")
    plt.ylabel("Number of words")
    plt.title("Average word count")

    plt.show()
    
def location_ttest(train):
    overall_mean = all_freq.count()/len(train.language)
    alpha = 0.05
    loc_cluster_one = Java_freq.count()/len(train[train.language == 'Java'])
    t, p = stats.ttest_1samp(loc_cluster_one, overall_mean)
    return t, p
    print(f'Test Statistic: {t.round(2)}, P-Value: {p.round(2)}')