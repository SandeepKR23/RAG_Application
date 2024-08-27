import streamlit as st
import sys
from src.RAGSystem import RAGSystem
from src.vectorStores import LocalVectorStore

def ask_ollama_question(question):
    if not hasattr(rag, "current_model") or rag.current_model != model_name:
        rag.load_from_ollama_models(model_name)
    res = rag.ask(question)
    return res

rag = RAGSystem()
lvs = LocalVectorStore()
lvs.load(folder_path="test")

rag.set_vector_db(lvs.vector_store)


st.title("LangChain Chatbot")

model_name = st.sidebar.selectbox(
    "Select Language Model",
    ("phi3.5", "llama3.1:8b", "phi3:3.8b", "gemma2:2b", "qwen2:0.5b", "qwen2:1.5b", "smollm:135m", "smollm:360m", "smollm:1.7b")
)

question = st.text_input("Enter your question:")

if st.button("Submit"):
    answer = ask_ollama_question(question)
    st.write("Response:")
    st.write(answer)
