import time
import gradio as gr
import requests
import json
from llama_cpp import Llama

llm = Llama(
    "../../../.cache/huggingface/hub/models--TheBloke--Mistral-7B-Instruct-v0.2-GGUF/snapshots/3a6fbf4a41a1d52e415a4958cde6856d34b2db93/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    # put YOUR MODEL HERE
)


def call_llama(prompt):
    stream = llm(
    prompt,
  max_tokens=850,
  temperature=0.0,
  stream=True
    )
    return stream

def send_message(message, history):
    generated = ''
    flag = True
    while flag:
        prompt = f"""<s>[INST] You're the most smartest expert in every field. Answer this question in a short manner and in the end of your response put END:\n ### Question:\n{message}\n Continue your response: ### Response:\n{generated}\n[/INST]"""      
        # generated += call_llama(prompt)
        stream = call_llama(prompt)
        for output in stream:
            out = json.dumps(output, indent=2)
            converted = json.loads(out)
            streamed_output = converted['choices'][0]['text']
            if generated.lower().__contains__('end'):
                flag=False
                break
            else:    
                generated += streamed_output
                yield generated
        if generated.__contains__('END'):
            flag=False

gr.ChatInterface(send_message).launch(share=True)














