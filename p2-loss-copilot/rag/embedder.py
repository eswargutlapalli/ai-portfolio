"""
Build Index function -
* Takes a list of strings called `documents`
* Splits them into chunks of 500 characters with 50 character overlap
* Builds and returns a FAISS vectorstore
Author: Eswar Gutlapalli
"""

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

def build_index(documents: list[str], embeddings):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents(documents)
    return FAISS.from_documents(chunks, embeddings)