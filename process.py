import pandas as pd
import numpy as np
import os, json
import nltk

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

