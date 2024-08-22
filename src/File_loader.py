from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
import logging
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from uuid import uuid4
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Retrieve API keys from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables")
if not PINECONE_INDEX_NAME:
    raise ValueError("PINECONE_INDEX_NAME not found in environment variables")

class DigitalDataLoader:
    def __init__(self) -> None:
        self._all_docs = []
        
    def get_text_chunks(self):
        """Splits the loaded documents into chunks of text."""
        try:
            if not self._all_docs:
                raise Exception("No documents loaded to split.")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1536, chunk_overlap=100,
                separators=["\n\n", "\n", "(?<=\. )", " ", ""],
            )
            chunks = text_splitter.split_documents(self._all_docs)
            print(f"Number of chunks split: {len(chunks)}")
            return chunks
        except Exception as e:
            print(f"Error occurred during text splitting: {str(e)}")
            raise e
        
    def set_pdf_files(self, pdf_path):
        """Loads PDF files and adds their content to the document list."""
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            self._all_docs.extend(documents)
            print(f"Number of documents loaded are: {len(documents)}")
        except Exception as e:
            print(f"File '{pdf_file}' not found: {e}")

class FileProcessor:
    @staticmethod
    def get_chunks(pdf_file):
        loader = DigitalDataLoader()

        try:
            loader.set_pdf_files(pdf_file)
            return loader.get_text_chunks()
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            raise e

class pineconeDB:
    def __init__(self) -> None:
        self.index = None
        self.embeddings = None

    def setup_db(self):
        """Set up Pinecone database and embedding model."""
        try:
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=PINECONE_API_KEY)

            # Check if the index exists and create if not
            if PINECONE_INDEX_NAME not in self.pc.list_indexes().names():
                self.pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=1536,  # Adjust dimension based on your embedding model
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-west-1'
                    )
                )
                
            self.index = self.pc.Index(PINECONE_INDEX_NAME)
            print(f"Index '{PINECONE_INDEX_NAME}' details: {self.index.describe_index_stats()}")
            self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

        except Exception as e:
            print(f"Error setting up Pinecone DB: {str(e)}")
            raise e

    def _get_vector_store(self):
        return PineconeVectorStore(index=self.index, embedding=self.embeddings)
    
    def embed_and_store(self, chunks):
        """Embed text chunks and store them in Pinecone."""
        try:
            chunk_texts = [chunk.page_content for chunk in chunks]
            metadata = [chunk.metadata for chunk in chunks]

            documents = [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(chunk_texts, metadata)
            ]

            uuids = [str(uuid4()) for _ in range(len(documents))]

            print("documents: ", documents)
                
            vector_store = self._get_vector_store()
            vector_store.add_documents(documents=documents, ids=uuids)

        except Exception as e:
            print(f"Error occurred while embedding and storing vectors: {str(e)}")
            raise e

    def vector_search(self, query):
        try:
            vector_store = self._get_vector_store()
            results = vector_store.similarity_search(query,k=1,)
            print(f"Search results: {results}")
            return results
        except Exception as e:
            print(f"Error occurred during vector search: {str(e)}")
            raise e

# Example of how to run the code
if __name__ == "__main__":
    try:
        base_path = os.path.expanduser("~/Documents/VS_code_projects/Sandeep/GenAICourse")
        pdf_file = os.path.join(base_path, "data", "Mock_Intuitive_report.pdf")
        print(f"The path is : {pdf_file}")
        chunks = FileProcessor.get_chunks(pdf_file)
        
        db = pineconeDB()
        db.setup_db()
        db.embed_and_store(chunks)

        # Example search query
        query = "What is the difference between a 1-bit and 2-bit LLM?"
        search_results = db.vector_search(query)
        print(f"Search results: {search_results}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
