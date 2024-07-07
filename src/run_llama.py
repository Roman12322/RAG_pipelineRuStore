import time
import gradio as gr
import requests
import json
from llama_cpp import Llama
from config import PINECONE_API
from langchain.llms import LlamaCpp
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain  
from run_vectorizing_data import init_db_index, init_embedding_model
import gradio as gr

def init_vectorstore(index, embedding_model, text_field):
    vectorstore = Pinecone(
    index, embedding_model.embed_query, text_field
    )
    return vectorstore


llm = Llama(
    "../../../.cache/huggingface/hub/models--TheBloke--Mistral-7B-Instruct-v0.2-GGUF/snapshots/3a6fbf4a41a1d52e415a4958cde6856d34b2db93/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_gpu_layers=30,
    n_ctx=1024
    # put YOUR MODEL HERE
)


def get_context(message):
    embed_model = init_embedding_model() # Initial embedding model with 384 dim_size
    
    pc, index  = init_db_index(index_name = 'rustore', api_key=PINECONE_API)
    vector_store = init_vectorstore(index, embed_model, text_field='question')
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    docs = retriever.get_relevant_documents(message)[0]
    returned_message = ""
    response = docs.metadata['response'].replace('\t',' ')
    return response

def call_llama(prompt):
    stream = llm(
    prompt,
  max_tokens=350,
  temperature=0.0,
  stream=True
    )
    return stream

def send_message(message, history):
        generated = ''
        flag = True
        context = get_context(message)  # Get context for code generation from documentation
        
        while flag:
            prompt = f"""<s>[INST] You're the most smartest expert in Java Programming. Generate function for this context and in the end of your response put <eot>:\n ### Question:\n{message}\n###Context:\n{context}\nContinue your response: ### Response:\n{generated}\n[/INST]"""      
            try:
                stream = call_llama(prompt)
                for output in stream:
                    out = json.dumps(output, indent=2)
                    converted = json.loads(out)
                    streamed_output = converted['choices'][0]['text']
                    if generated.lower().__contains__('<eot>'):
                        flag=False
                        break
                    else:    
                        generated += streamed_output
                        yield generated
                if generated.__contains__('<eot>'):
                    flag=False
            except:
                break
                    

gr.ChatInterface(send_message).launch(share=True)














