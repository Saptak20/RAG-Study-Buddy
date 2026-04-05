from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    return splitter.split_documents(documents)


def create_vector_store(docs, embeddings):
    return FAISS.from_documents(docs, embeddings)


def get_retriever(vector_db, k=3):
    return vector_db.as_retriever(search_kwargs={"k": k})
