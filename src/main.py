from RAGSystem import RAGSystem
from vectorStores import LocalVectorStore
class Main:
    def ask_question_from_local_stores(folder_path, question):
        rag = RAGSystem()
        lvs = LocalVectorStore()
        lvs.load(folder_path)

        rag.set_vector_db(lvs.vector_store)
        rag.load_ollama()
        res = rag.ask(question)
        return res
    def get_rag(files: list[str]):
        rag = RAGSystem()
        lvs = LocalVectorStore()
        for f in files:
            if f.endswith(".pdf"):
                lvs.add_pdf(f)
            elif  f.endswith(".txt"):
                lvs.add_text_file(f)
            else:
                lvs.add_content(f)
        
        rag.set_vector_db(lvs.vector_store)
        rag.load_ollama()
        return rag, lvs