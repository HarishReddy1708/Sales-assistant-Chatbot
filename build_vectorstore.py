# build_vectorstore.py
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import os


def load_docs():
    raw_texts = []
    for fname in os.listdir("docs"):
        if fname.endswith(".txt"):
            with open(os.path.join("docs", fname), "r", encoding="utf-8") as f:
                raw_texts.append(f.read())
    return raw_texts


def build_vectorstore():
    texts = load_docs()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = [
        Document(page_content=chunk)
        for text in texts
        for chunk in splitter.split_text(text)
    ]

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("faiss_index")
    print("Vector store saved to faiss_index/")


if __name__ == "__main__":
    build_vectorstore()
