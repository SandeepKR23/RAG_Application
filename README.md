# RAG_Application
RAG application with Langchain Ollama
=======
# ML Chatbot Project

# Created a environment
```
conda create -p venv python==3.11

conda activate venv/
```
# Install all necessary libraries
```
pip install -r requirements.txt
```

# Commands to execute code
```
To run locally:

uvicorn app:app --host 0.0.0.0 --port 5000 --ssl-keyfile "C:/Users/rajku-sa/Documents/MEGA/Ineuron/Chatbot_Pro/certificates/domain.key" --ssl-certfile "C:/Users/rajku-sa/Documents/MEGA/Ineuron/Chatbot_Pro/certificates/domain-1689779644033.crt"
```
# Run in dockers:
```
Set this in app.py
uvicorn.run(app, host="0.0.0.0", port=5000)

Run in command prompt
uvicorn app:app --host 0.0.0.0 --port 5000

```

# Postman API's with data:

```
https://192.168.253.122:5000/create-chatbot
{
  "Chatbot_Name": "Chatbot123",
  "files": [
    {
      "filename": "ECT 01 Composable Enterprises Intro.pdf"
    },
    {
      "filename": "Research in AI-based chatbot.pdf"
    }
  ]
}
```

# File server path 
```
Local path for testing
FILE_PATH_PREFIX3='C:/Users/rajku-sa/Documents/MEGA/Ineuron/ML_Chatbot_Project/artifacts/'

Dockers server path:
FILE_PATH_PREFIX='/opt/ect/data/lcap/apps/3275/files/'
```
