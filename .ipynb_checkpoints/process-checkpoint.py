import pandas as pd
import numpy as np
import os, json, nltk

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

def extract_simple_docs(df):
    docs = []
    for row in df.iterrows():
        abstract = row[1]['abstract']
        title = row[1]['title']
        if len(title) < 20:
            title = ''
        doc = title + ' ' + abstract
        docs.append(doc)
    
    return docs

def get_stopwords():
    languages = ['english', 'spanish', 'french']
    stop_words = []
    for lang in languages:
        stop_words.extend(stopwords.words(lang))
    
    return set(stop_words)

def simple_preprocess(docs):
    new_docs = []
    stop_words = get_stopwords()
    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(docs)):
        new_doc = docs[idx].lower()  # Convert to lowercase.
        new_doc = tokenizer.tokenize(new_doc)  # Split into words.
        new_doc = [token for token in new_doc if not token.isnumeric()]
        new_doc = [token for token in new_doc if len(token) > 1]
        new_doc = [token for token in new_doc if token not in stop_words]
        new_docs.append(new_doc)
    
    return new_docs

# def make_lda(dictionary, corpus, num_topics, df):
    