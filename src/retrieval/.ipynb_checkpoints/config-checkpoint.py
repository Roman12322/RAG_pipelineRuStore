import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API=os.getenv('PINECONE_API')
# print(PINECONE_API)
