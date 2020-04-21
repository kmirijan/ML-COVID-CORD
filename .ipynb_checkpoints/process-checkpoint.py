import pandas as pd
import numpy as np
import os, json, nltk

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from gensim.models import LdaMulticore

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

def make_lda(dictionary, corpus, num_topics):
    passes = 10

    # Make a index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    model = LdaMulticore(
        corpus=corpus,
        id2word=id2word,
        passes=passes,
        num_topics=num_topics
    )
    
    return model

def get_model_stats(model, docs, dictionary, num_topics, verbose=False):
    top_topics = model.top_topics(texts=docs, dictionary=dictionary, coherence='c_v') #, num_words=20)

    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    rstd_atc = np.std([t[1] for t in top_topics]) / avg_topic_coherence
  
    if verbose:
        print('Average topic coherence: ', avg_topic_coherence)
        print('Relative Standard Deviation of ATC: ', rstd_atc)
        
    return avg_topic_coherence, rstd_atc
    