from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from torch import cuda
from datasets import load_dataset
from pinecone import Pinecone
from config import PINECONE_API
from transformers import pipeline
import pandas as pd
import numpy as np
import json
import os

def set_nan_as_empty(row):
    if isinstance(row, float):
        return 'empty'
    else:
        return row

def read_dataset():
    df = pd.read_excel("Книга1.xlsx")
    df.columns = ['question', 'response', 'link']
    df['link'] = df['link'].apply(set_nan_as_empty)
    return df

def init_embedding_model():
    embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    embed_model = HuggingFaceEmbeddings(
        model_name=embed_model_id,
        model_kwargs={'device': device},
        encode_kwargs={'device': device, 'batch_size': 32}
    )
    return embed_model

def upsert_data(index, data, embedding_model):
    batch_size = 4

    for i in range(0, len(data), batch_size):
        i_end = min(len(data), i+batch_size)
        batch = data.iloc[i:i_end]
        ids = [str(idx) for idx in range(i, i_end)] # TODO set some row_id for each chunk
        print(f"ids: {ids}\n")
        texts = [x['question'] for i, x in batch.iterrows()] # TODO extract chunk of text from dataframe 
        embeds = embedding_model.embed_documents(texts)
        metadata = [
            {'response': x['response'],
            'link': x['link'],
            'question': x['question']} for i, x in batch.iterrows()
        ] # TODO set OWN METADATA
        index.upsert(vectors=zip(ids, embeds, metadata))


def init_db_index(index_name, api_key):
    pc = Pinecone(api_key)
    index = pc.Index(index_name)
    print(index.describe_index_stats())
    
    return pc, index


index_name = 'rustore'
pc, index = init_db_index(index_name=index_name, api_key=PINECONE_API) # Initial index for Pinecone Vector store
embed_model = init_embedding_model() # Initial embedding model with 384 dim_size

# # UPSERT DATA TO OUR VECTOR_STORE
# data = read_dataset()
# upsert_data(index, data, embed_model)
















