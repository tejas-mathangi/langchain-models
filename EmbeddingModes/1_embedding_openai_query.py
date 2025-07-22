from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model ="text-embedding-3-small", dimensions = 32 ) 

result = embedding.embed_query("Delhi is capital of India")

print(str(result))

# generating embeddings for a single query
