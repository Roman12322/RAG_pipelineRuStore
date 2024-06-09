from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="RichardErkhov/unsloth_-_Hermes-2-Pro-Mistral-7B-gguf",
    filename="Hermes-2-Pro-Mistral-7B.Q4_K_S.gguf",
    verbose=False
)
