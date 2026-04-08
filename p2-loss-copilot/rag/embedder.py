"""
Build Index function -
* Takes a list of strings called `documents`
* Splits them into chunks of 500 characters with 50 character overlap
* Builds and returns a FAISS vectorstore
Author: Eswar Gutlapalli
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def build_index(documents: list[str]) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

if __name__ == "__main__":
    sample_docs = [
        "Midwest region showed 12% increase in auto loan defaults in Q3.",
        "Commercial real estate losses concentrated in urban markets.",
        "Credit card delinquencies rose among 25-34 age group."
    ]
    print("Building index...")
    index = build_index(sample_docs)
    print("Index built successfully:", type(index))