import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nltk
import unicodedata
import re
import requests


import nltk.sentiment
import matplotlib.pyplot as plt
import seaborn as sns

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
    
