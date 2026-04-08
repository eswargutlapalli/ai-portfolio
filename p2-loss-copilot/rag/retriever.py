"""
Get Relevant Chunks function -
- Takes a query and a vectorstore (already loaded FAISS index)
- Returns the top-k most similar chunks to the query
Author: Eswar Gutlapalli
"""

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def get_relevant_chunks(query: str, vectorstore: FAISS, k: int = 3) -> list:
    return vectorstore.similarity_search(query, k=k)

if __name__ == "__main__":
    from embedder import build_index

    sample_docs = [
        "Midwest region showed 12% increase in auto loan defaults in Q3.",
        "Commercial real estate losses concentrated in urban markets.",
        "Credit card delinquencies rose among 25-34 age group."
    ]

    print("Building index...")
    index = build_index(sample_docs)

    print("Searching...")
    results = get_relevant_chunks("What are the loan defaults?", index)

    for i, chunk in enumerate(results):
        print(f"\nChunk {i+1}: {chunk.page_content}")