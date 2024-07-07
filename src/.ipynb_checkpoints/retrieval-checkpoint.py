from config import PINECONE_API
from langchain.llms import LlamaCpp
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain  
from run_vectorizing_data import init_db_index, init_embedding_model
from config import PINECONE_API
import gradio as gr

def init_vectorstore(index, embedding_model, text_field):
    vectorstore = Pinecone(
    index, embedding_model.embed_query, text_field
    )
    return vectorstore

def query(message, history):
    
    embed_model = init_embedding_model() # Initial embedding model with 384 dim_size
    
    pc, index  = init_db_index(index_name = 'rustore', api_key=PINECONE_API)
    vector_store = init_vectorstore(index, embed_model, text_field='question')
    
    # rag = init_langchain_retrieval(vector_store)
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    docs = retriever.get_relevant_documents(message)[0]
    returned_message = ""
    response = docs.metadata['response'].replace('\t',' ')
    
    for letter in response:
        returned_message += letter
        yield returned_message
        
    if docs.metadata['link'] == 'empty':
        pass
    else:    
        for letter in '\nLink: ':
            returned_message += letter
            yield returned_message
        for letter in docs.metadata['link']:
            returned_message += letter
            yield returned_message

gr.ChatInterface(query).launch(share=True)




