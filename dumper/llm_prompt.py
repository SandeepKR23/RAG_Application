from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import sys
import os
from pinecone import Pinecone
from pineconeD import Pinecone_Vector_DB

# Ensure environment variables are loaded
load_dotenv()

class ChatBot():
    def __init__(self, model):
        p_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.pinecone_db = Pinecone_Vector_DB(p_client)
        self.vectorstore = self.pinecone_db.get_vectorstore()
        self.model = model

    def vector_store(self):
        # Build prompt
        template = """Please utilize the provided context data below to formulate a well-informed response to the subsequent question. 
                        Should the answer be beyond your current knowledge scope, kindly indicate your lack of information transparently without 
                        attempting to speculate an answer. Aim for precision and succinctness in your reply, limiting it to a maximum of three sentences..

            {context}
            Question:{question}

            Your Response:
                Please provide your answers in the specified format:
                'Answer : [Insert the answer here]'
            AI bot Answer: 
        """
        print(f"Model: {self.model}")
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        qa_chain= RetrievalQA.from_chain_type(
                llm = self.model, 
                chain_type="stuff", 
                retriever = self.vectorstore.as_retriever(), 
                chain_type_kwargs= {"prompt": QA_CHAIN_PROMPT})

        return qa_chain, self.vectorstore 
    
        # Ask Questions
    def get_answer(self, qa_chain, question, vectorstore1):
        try: 
            #Initialize new variable for metadata listing
            meta = []

            response = qa_chain({"query": question})
            docs = vectorstore1.similarity_search(question, k=2)


            print("DOCS: ", len(docs))


            for i in range(len(docs)):  # Changed to len(docs) for safety
                meta.append(f"Page No: {docs[i].metadata.get('page', 'N/A')}, PDF URL: {docs[i].metadata.get('source', 'N/A')}, Published Date: {docs[i].metadata.get('published_date', 'N/A')}")
            
            
            return {"response": response["result"], "metadata": meta}
        
        except Exception as e:
            print(f"Error occurred during question processing: {str(e)}")
            raise e

    
class Question_main:
    def ask_question(question, model):
        visitor_obj = ChatBot(model)
        qa_chain, vectorstore = visitor_obj.vector_store()
        answer = visitor_obj.get_answer(qa_chain, question, vectorstore)
        return answer
    

# # Example usage
# if __name__ == "__main__":
#     question = "What is sustainalillity"
#     result = Question_main.ask_question(question)
#     print(result)
