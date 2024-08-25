from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader

class LocalVectorStore:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", embedding_dim=384):
        self.embedding_dim = embedding_dim
        self.documents = [] 
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1536, chunk_overlap=100,
            separators=[r"\n\n", r"\n", r"(?<=\. )", " ", ""],
        )
        self._db = None
    def _add_and_store(self, docs):
        if self._db is None:
            self.documents.extend(docs)
        else:
            splits = self.text_splitter.split_documents(docs)
            self._db.add_documents(new_splits)
    def add_pdf(self, pdf_path: str):
        self._add_and_store(PyPDFLoader(pdf_path).load())
    def add_csv_file(self, csv_file_path):
        self._add_and_store(CSVLoader(csv_file_path).load())
    def add_text_file(self, text_file_path: str):
        self._add_and_store(TextLoader(text_file_path).load())
    def add_content(self, content: str):
        self._add_and_store(Document(page_content=content))
    @property
    def vector_store(self):
        if self._db is None:
            splits = self.text_splitter.split_documents(self.documents)
            self._db = FAISS.from_documents(splits, self.embeddings)
        return self._db
    
    def save(self, folder_path: str):
        """ it requires folder name not file name"""
        self.vector_store.save_local(folder_path)
    def load(self, folder_path: str):
        self._db = FAISS.load_local(folder_path, self.embeddings, allow_dangerous_deserialization=True)