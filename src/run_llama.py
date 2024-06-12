import time
import gradio as gr
import requests
import json
from llama_cpp import Llama

llm = Llama(
    "../../../.cache/huggingface/hub/models--TheBloke--Mistral-7B-Instruct-v0.2-GGUF/snapshots/3a6fbf4a41a1d52e415a4958cde6856d34b2db93/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
)

# def main(message, history):
#     func_tests = pd.read_csv('func_test.csv')
#     flag = True
#     gen = ''
#     stop_seq='<|EOS|>'
#     while flag:
#         response, prompt_len = call_vllm(func_tests=func_tests, max_tokens_n=550, temp=0, generated=gen, user_message=message)
#         for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
#             if chunk:
#                 data = json.loads(chunk.decode("utf-8"))
#                 output = data['text'][0][prompt_len:]

#                 print(f"####output: \n{output}")
#                 if output.__contains__('<|EOT|>'):
#                     yield "\n" + output[:prompt_len-len(stop_seq)] + "\n"
#                     flag = False
#                     break
                
#                 else:
#                     gen+=output
#                     yield "\n" + output[:prompt_len] + "\n"
#             else:
#                 flag = False    
#                 print(f"##########################\n{gen}")


# chatbot = gr.Chatbot(height=600)
# chat_interface = gr.ChatInterface(main, chatbot=chatbot, analytics_enabled=True)
# with gr.Blocks() as demo:
#     chat_interface.render()

#demo.queue(default_concurrency_limit=100).launch(server_name="0.0.0.0", server_port=8999, ssl_keyfile='/workspace/s.key', ssl_certfile='/workspace/s.cer', ssl_verify=False, show_api=False) 
# demo.queue(default_concurrency_limit=100).launch(server_name="10.1.126.11", server_port=9954, ssl_verify=False, show_api=False)


def call_llama(prompt):
    # headers = {
    #     'Content-Type': 'application/json',
    # }
    stream = llm(
    prompt,
  max_tokens=550,
  temperature=0.0,
  stream=True
    )
    # json_data = {
    #       "prompt": prompt,
    #       "max_new_tokens": 550,
    #       "temperature": 0.0,
    #       'stream': True
    # }
    # response = requests.post('http://localhost:9900/v1/completions', headers=headers, json=json_data, stream=True)
    # response_dict = json.loads(response.text) 
    # resp_text = response_dict['choices'][0]['text']
    # return resp_text
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
        # if converted['choices'][0]['text'].__contains__('END'):
            flag=False
        # print(generated)
        # print(f"\n\n########## PROMPT{prompt}\n\n")
    # yield generated.split("END")[0]

gr.ChatInterface(send_message).launch(share=True)














