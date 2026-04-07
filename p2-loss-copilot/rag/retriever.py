"""
Get Relevant Chunks function -
- Takes a query: str and a vectorstore (already loaded FAISS index) and k: int = 3 as parameters
- Returns the top-k most similar chunks to the query
Author: Eswar Gutlapalli
"""

def get_relevant_chunks(query: str, vectorstore, k: int = 3):
    return vectorstore.similarity_search(query, k)