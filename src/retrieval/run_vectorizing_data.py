from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from torch import cuda
from datasets import load_dataset
from pinecone import Pinecone
from config import PINECONE_API
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.chains import RetrievalQA
import pandas as pd
import numpy as np
import json
import os

def read_dataset(path):
    with open(path, 'r') as f:
        data = f.read()
        data = json.loads(data)
    
    dataframe = pd.DataFrame(data['data'])
    qna = dataframe[['title', 'description', 'parent_url']]
    qna = qna.rename(columns={'title':'question','description': 'answer','parent_url': 'link'})
    print(qna.head(3))
    return qna

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
        texts = [x['answer'] for i, x in batch.iterrows()] # TODO extract chunk of text from dataframe 
        embeds = embedding_model.embed_documents(texts)
        # print(f"shape of embeddings: {np.asarray(embeds).shape}")
        # print(f"texts: \n {texts}")
        # get metadata to store in Pinecone
        metadata = [
            {'answer': x['answer'],
            'source': x['link'],
            'question': x['question']} for i, x in batch.iterrows()
        ] # TODO set OWN METADATA
        # print(f"\metadata: \n {metadata}")
        index.upsert(vectors=zip(ids, embeds, metadata))


def init_db_index(index_name, api_key):
    pc = Pinecone(api_key)
    index = pc.Index(index_name)
    print(index.describe_index_stats())
    
    return pc, index


# index_name = 'tinkoff'
# pc, index = init_db_index(index_name=index_name, api_key=PINECONE_API) # Initial index for Pinecone Vector store
# embed_model = init_embedding_model() # Initial embedding model with 384 dim_size

# # UPSERT DATA TO OUR VECTOR_STORE
# data = read_dataset("../data/dataset.json")
# upsert_data(index, data, embed_model)
















