from config import PINECONE_API
from langchain.llms import LlamaCpp
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain  
from run_vectorizing_data import init_db_index, init_embedding_model
from config import PINECONE_API
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import warnings
from pprint import pprint
import gradio as gr

warnings.filterwarnings('ignore')

def init_vectorstore(index, embedding_model, text_field):
    vectorstore = Pinecone(
    index, embedding_model.embed_query, text_field
    )
    return vectorstore
    
def build_llm():
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
            model_path='../../../../.cache/huggingface/hub/models--TheBloke--Mistral-7B-Instruct-v0.2-GGUF/snapshots/3a6fbf4a41a1d52e415a4958cde6856d34b2db93/mistral-7b-instruct-v0.2.Q4_K_M.gguf', 
            temperature=0.0,
            f16_kv=True,
            max_tokens = 250,
            n_ctx=8000,
            n_gpu_layers=1,
            n_batch=250,
            verbose=False, 
            top_p=0.75,
            top_k=40,
            repetition_penalty=1.1,
            callback_manager = callback_manager
    )
    return llm
    
def init_langchain_retrieval(vectorstore):
    # Build LLM for our Retriever
    llm = build_llm()
    # Init pipeline 
    rag_pipeline = RetrievalQA.from_chain_type(
    llm=llm, chain_type='stuff',
    retriever=vectorstore.as_retriever(),
    verbose=True,
    return_source_documents=True
    )
    return rag_pipeline

def query(message, history):
    
    embed_model = init_embedding_model() # Initial embedding model with 384 dim_size
    
    pc, index  = init_db_index(index_name = 'tinkoff', api_key=PINECONE_API)
    vector_store = init_vectorstore(index, embed_model, text_field='question')
    
    rag = init_langchain_retrieval(vector_store)

    response = rag(message)
    final = ""
    for item in response['result']:
        final+= item
        yield final

gr.ChatInterface(query).launch(share=True)




