from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model ="text-embedding-3-small", dimensions = 32 ) 

documents = [
    "Delhi is the capital of India.",
    "The capital of France is Paris.",
    "Tokyo is the capital of Japan."
]
result = embedding.embed_documents(documents)

print(str(result))

# generating embeddings for multiple documents/sentences/queries/paragraphs/texts..

