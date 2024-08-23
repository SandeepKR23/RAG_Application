import os
import faiss
import numpy as np
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

class LocalVectorStore:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", embedding_dim=384):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)  # L2 distance index
        self.texts = []  # Store texts associated with the vectors
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1536, chunk_overlap=100,
            separators=[r"\n\n", r"\n", r"(?<=\. )", " ", ""],
        )

    def add_embeddings(self, text: str):
        chunks = self.text_splitter.split_text(text)
        vectors = [self.embeddings.embed_query(chunk) for chunk in chunks]
        vectors_np = np.array(vectors).astype('float32')
        self.index.add(vectors_np)
        self.texts.extend(chunks)
    
        
    def add_pdf(self, pdf_path: str):
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

        self.add_embeddings(text)

    def add_text_file(self, text_file_path: str):
        with open(text_file_path, 'r') as file:
            text = file.read()
        self.add_embeddings(text)

    def add_content(self, content: str):
        self.add_embeddings(content)

    def search(self, query: str, k=5):
        query_vector = np.array(self.embeddings.embed_query(query)).astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_vector, k)
        results = [(self.texts[i], distances[0][idx]) for idx, i in enumerate(indices[0])]
        return results