from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") 

text = "This is a sample text for embedding."

documents = [
    "Delhi is the capital of India.",
    "The capital of France is Paris.",
    "Tokyo is the capital of Japan."
]

vector = embedding.embed_query(text)

result = embedding.embed_documents(documents)

print(str(vector))
print(str(result)) 