from unicodedata import normalize
import requests
import regex as re
import datetime
from scipy.stats import randint, loguniform
import pandas as pd

import pandas_profiling
from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, precision_recall_fscore_support

from sklearn.naive_bayes import MultinomialNB
from lightgbm import LGBMClassifier


def helper_function():
    print("hello_world")
def train_df_path():
    return r'./data/train_df.csv'


def tratamentos(df):
    def retorna_stopwords():
        urlstopwords=r"https://gist.githubusercontent.com/alopes/5358189/raw/2107d809cca6b83ce3d8e04dbd9463283025284f/stopwords.txt"
        r=requests.get(urlstopwords)
        list_stopwords=list(map(lambda x: x.replace(' ',''),r.text.split('\n')))
        return list_stopwords
    
    def remover_stop_words(x,list_stopwords):
        x=[word for word in x.split() if word not in list_stopwords]
        x=' '.join(x)
        return x
    
    listastopwords=retorna_stopwords()
    
    df=df.drop(['submission_date','reviewer_id','product_id', 'product_brand'],axis=1)
    
    for coluna in df.columns:
        
        if df[coluna].dtype=='object':
            #Input de missings, para alta cardinalidade inputo a palavra vazio e baixa cardinalidade a moda
            if df[coluna].nunique()>500:
                df[coluna]=df[coluna].fillna('vazio')
            else:
                df[coluna]=df[coluna].fillna(df[coluna].mode()[0])
            #Remove caracteres especiais e acentuação
            df[coluna]=df[coluna].str.normalize('NFKD').str.encode('ASCII','ignore').str.decode('ASCII')
            df[coluna]=df[coluna].map(lambda x : re.sub(r'[^\w\s]', '', x))
            #Textos para minúscula
            df[coluna]=df[coluna].str.lower()
            #Remove stop words
            df[coluna]=df[coluna].map(lambda x: remover_stop_words(x,listastopwords))
            #Transforma números na tag NUM
            df[coluna]=df[coluna].apply(lambda x: re.sub(r'\d+', 'NUM', x))
        else:
            df[coluna]=df[coluna].fillna(df[coluna].median())


    def trataidade(colunaano):
        return datetime.datetime.now().year-colunaano

    df.reviewer_birth_year=trataidade(df.reviewer_birth_year)

    df=df.rename(columns={'reviewer_birth_year':'age'})

    def tratarecommend(colunarecommend):
        return pd.get_dummies(colunarecommend,prefix='recommend_to_a_friend',drop_first  =True)

    df.recommend_to_a_friend=tratarecommend(df.recommend_to_a_friend)

    def tratargender(colunagender):
        return pd.get_dummies(colunagender,prefix='reviewer_gender',drop_first  =True)

    df.reviewer_gender=tratargender(df.reviewer_gender)

    def tratarstate(colunastate):
        return pd.get_dummies(colunastate,prefix='reviewer_state')

    df=pd.concat([df,tratarstate(df.reviewer_state)],axis=1)

    df=df.drop(['reviewer_state'],axis=1)
    
    return df


def treino_teste(dfsplit):
    dftreino=dfsplit[dfsplit.submission_date<'2018-04-01'].reset_index(drop=True)
    dfteste=dfsplit[dfsplit.submission_date>='2018-04-01'].reset_index(drop=True)
    dftreino=tratamentos(dftreino)
    dfteste=tratamentos(dfteste)
    return dftreino.drop('overall_rating',axis=1), dfteste.drop('overall_rating',axis=1),dftreino['overall_rating'], dfteste['overall_rating']