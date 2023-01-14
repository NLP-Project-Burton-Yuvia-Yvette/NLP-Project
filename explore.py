import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nltk
import unicodedata
import re
import requests

from wordcloud import WordCloud
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
    
    #generate mean of sentiment_score by period
  
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

    ###################### word cloud ###################################
def create_subgroups (train):    
    javaScript_words = ' '.join(train[train.language == 'JavaScript'].clean_text).split(' ')
    java_words = ' '.join(train[train.language == 'Java'].clean_text).split(' ')
    python_words = ' '.join(train[train.language == 'Python'].clean_text).split(' ')
    typeScript_words = ' '.join(train[train.language == 'TypeScript'].clean_text).split(' ')
    all_words = ' '.join(train.clean_text).split(' ')
    return javaScript_words, java_words, python_words, typeScript_words, all_words


def get_frequency(javaScript_words, java_words, python_words, typeScript_words, all_words ):
    JavaScript_freq = pd.Series(javaScript_words).value_counts()
    Java_freq = pd.Series(java_words).value_counts()
    Python_freq = pd.Series(python_words).value_counts()
    TypeScript_freq = pd.Series(typeScript_words).value_counts()
    All_words_freq = pd.Series(all_words).value_counts()
    return JavaScript_freq,Java_freq,Python_freq, TypeScript_freq, All_words_freq

def create_wordcounts(JavaScript_freq,Java_freq,Python_freq, TypeScript_freq, All_words_freq ):
    word_counts = (pd.concat([JavaScript_freq, Java_freq, Python_freq, TypeScript_freq, All_words_freq], axis=1, sort=True)
                .set_axis(['JavaScript', 'Java', 'Python', 'TypeScript', 'AllWords'], axis=1, inplace=False)
                .fillna(0)
                .apply(lambda s: s.astype(int)))
    word_counts['raw_count'] = word_counts.AllWords
    word_counts['frequency'] = word_counts.raw_count / word_counts.raw_count.sum()
    word_counts['augmented_frequency'] = word_counts.frequency / word_counts.frequency.max()
    return word_counts



def get_wordcloud(word_counts):
    
    # prepare words for wordcloud
    top_words_cloud = word_counts.sort_values(by='AllWords', ascending=False).head(50)
    top_words_cloud= top_words_cloud.index.to_list()
    top_words_cloud = " ".join(top_words_cloud)



    img = WordCloud(background_color='white',colormap='Accent').generate(top_words_cloud)
    # WordCloud() produces an image object, which can be displayed with plt.imshow
    plt.imshow(img)
    # axis aren't very useful for a word cloud
    plt.axis('off')
    return plt.show()

##################################### top 20 words #################################
def get_bigrams_graphs(python_words, javaScript_words):
    
    top_20_Python_bigrams = (pd.Series(nltk.ngrams(python_words, 2))
                          .value_counts()
                          .head(20))

    top_20_Python_bigrams.head()

    top_20_Python_bigrams.sort_values(ascending=False).plot.barh(colormap='Accent', width=.9, figsize=(10, 6))

    plt.title('20 Most frequently occuring Python bigrams')
    plt.ylabel('')
    plt.xlabel('# Word Count')

    # make the labels pretty
    ticks, _ = plt.yticks()
    labels = top_20_Python_bigrams.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)
    plt.show()


    top_20_JavaScript_bigrams = (pd.Series(nltk.ngrams(javaScript_words, 2))
                          .value_counts()
                          .head(20))
    top_20_JavaScript_bigrams.sort_values(ascending=False).plot.barh(colormap='Accent', width=.9, figsize=(10, 6))

    plt.title('20 Most frequently occuring JavaScript bigrams')
    plt.ylabel('')
    plt.xlabel('# Word Count')

    # make the labels pretty
    ticks, _ = plt.yticks()
    labels = top_20_JavaScript_bigrams.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)
    plt.show()

def get_bigrams(words, n):
    top_20_bigrams = (pd.Series(nltk.ngrams(words, n))
                    .value_counts()
                    .head(20))
    return top_20_bigrams

