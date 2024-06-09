import time
import gradio as gr
import requests
import json

def call_llama(prompt):
    headers = {
        'Content-Type': 'application/json',
    }
    
    json_data = {
        "prompt": prompt,
          "stop": [
          ],
          "max_new_tokens": 550,
          "temperature": 0.0,
          # "top_k": 100
    }
    response = requests.post('http://localhost:8000/v1/completions', headers=headers, json=json_data)
    response_dict = json.loads(response.text) 
    resp_text = response_dict['choices'][0]['text']
    return resp_text

def send_message(message, history):
    generated = ''
    flag = True
    while flag:
        prompt = f"""<s>[INST] You're the most smartest finance expert. Answer this question in a short manner and in the end of your response put END:\n ### Question:\n{message}\n Continue your response: ### Response:\n{generated}\n[/INST]"""
        generated += call_llama(prompt)
        yield generated
        if generated.__contains__('END'):
            flag=False
        print(generated)
        print(f"\n\n########## PROMPT{prompt}\n\n")
    yield generated.split("END")[0]

gr.ChatInterface(send_message).launch(share=True)