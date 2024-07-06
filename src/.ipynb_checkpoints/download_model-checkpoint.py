from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    verbose=False
)
