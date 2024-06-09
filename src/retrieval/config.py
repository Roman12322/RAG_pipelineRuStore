import os
from dotenv import load_dotenv

load_dotenv()

TG_BOT_TOKEN=os.getenv('TG_BOT_TOKEN')
PINECONE_API=os.getenv('PINECONE_API')
