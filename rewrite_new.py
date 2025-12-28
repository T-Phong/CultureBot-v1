from groq import Groq
from reranking import advanced_search
import os
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set. Please create .env file with GROQ_API_KEY=your_key")

client = Groq(api_key=groq_api_key)

# Rest of your rewrite.py code here...
