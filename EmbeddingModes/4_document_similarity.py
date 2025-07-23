# basic semantic search 
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

document = [
    "Delhi is the capital of India.",
    "The capital of France is Paris.",
    "Tokyo is the capital of Japan."
]

query = "What is the capital of Japan?"

query_embedding = embedding.embed_query(query)
doc_embedding = embedding.embed_documents(document)

scores = cosine_similarity([query_embedding], doc_embedding)[0]

index , scores = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(f"Query: {query}")
print(f"Most similar document: {document[index]}")
print(f"Similarity score: {scores}")

# This code performs semantic search by embedding a query and documents, then calculating the cosine similarity to find the most relevant document. 
# The HuggingFaceEmbeddings class is used to generate embeddings for both the query and the documents, and the sklearn library is used to compute cosine similarity. The
# result is the document that is most similar to the query along with its similarity score.

# Output:
#   % python 3.EmbeddingModels/document_similarity.py 
# Query: What is the capital of India?
# Most similar document: Delhi is the capital of India.
# Similarity score: 0.7655412901368366
#   % python 3.EmbeddingModels/document_similarity.py 
# Query: What is the capital of Japan?
# Most similar document: Tokyo is the capital of Japan.
# Similarity score: 0.8588654955020186
