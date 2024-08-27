from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
import torch
class RAGSystem:
    def set_vector_db(self, db):
        self.vector_store = db
    def ask(self, query):
        result = self.qa_chain({"query": query})
        return {
            "answer": result['result'],
            "source_documents": [doc.page_content for doc in result['source_documents']]
        }
    def load_model_phi(self):
        model_name = "microsoft/phi-1_5"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        pipe = pipeline("text-generation", model=self.model, tokenizer=tokenizer, 
                       device=device.index if device.type == "cuda" else -1,
                        max_new_tokens=512,  # Adjust this value as needed
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.95,
                        return_full_text=False
                       )
        self.llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={"temperature": 0.2, "max_length": 2048})
        retriever = self.vector_store.as_retriever()
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
    def load_model_phi_mini_instruct(self):
        # not able to run because it takes more memory
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3.5-mini-instruct", 
            device_map="cuda", 
            torch_dtype="auto", 
            trust_remote_code=True, 
        )
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=tokenizer,
        )
        generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }
        
        self.llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={"temperature": 0.7, "max_length": 2048})
        retriever = self.vector_store.as_retriever()
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
    def load_ollama(self):
        if hasattr(self, "current_model") and self.current_model == "llama3.1:8b":
            return
        from langchain.llms import Ollama
        self.llm = Ollama(model="llama3.1:8b")
        self.current_model = "llama3.1:8b"
        retriever = self.vector_store.as_retriever()
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
    def load_from_ollama_models(self, name= "llama3.1:8b"):
        if hasattr(self, "current_model") and self.current_model == name:
            return
        from langchain_community.llms import Ollama
        self.llm = Ollama(model=name)
        self.current_model = name
        retriever = self.vector_store.as_retriever()
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )