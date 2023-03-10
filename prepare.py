import unicodedata
import re
import json
# nltk, tokenization, stopwords
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
# pandas dataframe manipulation, acquire script, time formatting
import pandas as pd
import acquire
from time import strftime
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')


def basic_clean(string):
    '''
    This function takes in a string and
    returns the string normalized.
    '''
    # we will normalize our data into standard NFKD unicode, feed it into an ascii encoding
    # decode it back into UTF-8
    string = unicodedata.normalize('NFKD', string)\
             .encode('ascii', 'ignore')\
             .decode('utf-8', 'ignore')
    # utilize our regex substitution to remove our undesirable characters, then lowercase
    string = re.sub(r"[^\w0-9'\s]", '', string).lower()
    return string


def tokenize(string):
    '''
    This function takes in a string and
    returns a tokenized string.
    '''
    # make our tokenizer, taken from nltk's ToktokTokenizer
    tokenizer = nltk.tokenize.ToktokTokenizer()
    # apply our tokenizer's tokenization to the string being input, ensure it returns a string
    string = tokenizer.tokenize(string, return_str = True)
    
    return string


def stem(string):
    '''
    This function takes in a string and
    returns a string with words stemmed.
    '''
    # create our stemming object
    ps = nltk.porter.PorterStemmer()
    # use a list comprehension => stem each word for each word inside of the entire document,
    # split by the default, which are single spaces
    stems = [ps.stem(word) for word in string.split()]
    # glue it back together with spaces, as it was before
    string = ' '.join(stems)
    
    return string


def lemmatize(string):
    '''
    This function takes in string for and
    returns a string with words lemmatized.
    '''
    # create our lemmatizer object
    wnl = nltk.stem.WordNetLemmatizer()
    # use a list comprehension to lemmatize each word
    # string.split() => output a list of every token inside of the document
    lemmas = [wnl.lemmatize(word) for word in string.split()]
    # glue the lemmas back together by the strings we split on
    string = ' '.join(lemmas)
    #return the altered document
    return string


def remove_stopwords(string, extra_words = ["'"], exclude_words = []):
    '''
    This function takes in a string, optional extra_words and exclude_words parameters
    with default empty lists and returns a string.
    '''
    # assign our stopwords from nltk into stopword_list
    stopword_list = stopwords.words('english')
    # utilizing set casting, i will remove any excluded stopwords
    stopword_list = set(stopword_list) - set(exclude_words)
    # add in any extra words to my stopwords set using a union
    stopword_list = stopword_list.union(set(extra_words))
    # split our document by spaces
    words = string.split()
    # every word in our document, as long as that word is not in our stopwords
    filtered_words = [word for word in words if word not in stopword_list]
    # glue it back together with spaces, as it was so it shall be
    string_without_stopwords = ' '.join(filtered_words)
    # return the document back
    return string_without_stopwords

def data_prep(df):
    '''
    data_prep function takes in a dataframe and drops all null values and removes untargetted languages and resets indes
    returns dataframe
    '''
    df.dropna(inplace=True)
    df = df[(df.language == 'Java') | (df.language=='JavaScript') | (df.language=='Python') | (df.language=='TypeScript')]
    df.reset_index(drop =True, inplace=True)
    return df

def text_prep(df):
    '''
    text_prep funciton takes in a dataframe and applies preps text for exploration and modeling
    returns dataframe
    '''
    df['clean_text']= df.readme_contents.apply(basic_clean)
    df['clean_text']= df.clean_text.apply(tokenize)
    df['clean_text']= df.clean_text.apply(lemmatize)
    df['clean_text']= df.clean_text.apply(remove_stopwords)
    
    return df

def split_data(df, target):
    '''
    split_date takes in a dataframe  and target variable and splits into train , validate, test 
    and stratifies on target variable
    
    The split is 20% test 80% train/validate. Then 30% of 80% validate and 70% of 80% train.
    Aproximately (train 56%, validate 24%, test 20%)
    
    returns train, validate, and test 
    '''
    # split test data from train/validate
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123, 
                                        stratify=df[target])

    # split train from validate
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123, 
                                   stratify=train_validate[target])

                                   
    return train, validate, test