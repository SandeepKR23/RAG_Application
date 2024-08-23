import streamlit as st
#from langchain.llms import OpenAI, DeepInfra, Ollama
from llm_prompt import ChatBot, Question_main  # Import the updated class and method
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

# Function to get the language model based on user selection
def get_model(model_name):
    if model_name == 'OpenAI':
        return ChatOpenAI(model_name="gpt-4", temperature=0)
    elif model_name == 'Groq':
        return ChatGroq(model="mixtral-8x7b-32768", temperature=0,)
    elif model_name == 'DeepInfra':
        return None
    else:
        return None


# Streamlit UI
st.title("LangChain Chatbot")

model_name = st.sidebar.selectbox(
    "Select Language Model",
    ("OpenAI", "Groq", "DeepInfra")  # Update this list based on available models
)

question = st.text_input("Enter your question:")

if st.button("Submit"):
    llm = get_model(model_name)
    if llm:
        # Use the selected model to get the answer
        answer = Question_main.ask_question(question, llm)
        st.write("Response:")
        st.write(answer)
    else:
        st.error("Selected model is not available.")

