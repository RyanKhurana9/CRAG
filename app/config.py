# app/config.py
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()

UPPER_TH = 0.7
LOWER_TH = 0.3

llm = ChatOllama(
    model="llama3.1:8b"
)

embedding = OllamaEmbeddings(
    model="nomic-embed-text"
)