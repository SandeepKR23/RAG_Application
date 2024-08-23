from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class Pinecone_Vector_DB:

    def __init__(self, p_client) -> None:
        self.p_client = p_client
    
    def pinecone_Index_status(self, PINECONE_INDEX_NAME):
        """
        Get the index status and details
        """
        status = self.p_client.describe_index(PINECONE_INDEX_NAME).status['ready']
        index = self.p_client.Index(PINECONE_INDEX_NAME)
        Index_data = index.describe_index_stats()

        return status, Index_data
    
    @staticmethod
    def get_vectorstore():

        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # Ensure the correct model is used
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  # Retrieve API key from environment variables
        PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "1bitllm")  # Default index name if not set

        # Create Pinecone vector store
        vectorstore = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME, 
            embedding=embeddings, 
            pinecone_api_key= PINECONE_API_KEY)

        return vectorstore
    
#     def vector_search(self, query, k=1):
#         try:
#             vectorstore = self.get_vectorstore()
#             results = vectorstore.similarity_search(query, k=k)
#             return results
#         except Exception as e:
#             print(f"Error occurred during vector search: {str(e)}")
#             raise

    
#     # Example of how to run the code
# if __name__ == "__main__":
#     try:
#         db = Pinecone_Vector_DB(p_client=Pinecone(api_key=os.getenv("PINECONE_API_KEY")))
        
#         # Checking the status of the index
#         status, index_data = db.pinecone_Index_status(os.getenv("PINECONE_INDEX_NAME", "1bitllm"))
        
#         # Example search query
#         query = "What is the difference between a 1-bit and 2-bit LLM?"
#         search_results = db.vector_search(query)
#         print(f"Search results: {search_results}")

#     except Exception as e:
#         print(f"An error occurred: {str(e)}")