from utils_rag import load_and_split_pdf, create_vector_store, get_retriever
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

os.environ["OPENAI_API_KEY"] = "your_api_key_here"

# Load + Split
docs = load_and_split_pdf("data/notes.pdf")

# Embeddings + Vector DB
embeddings = OpenAIEmbeddings()
vector_db = create_vector_store(docs, embeddings)

# Retriever
retriever = get_retriever(vector_db)

# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

print("📚 RAG Study Buddy Ready!\n")

while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break

    result = qa_chain(query)

    print("\n🤖 Answer:", result["result"])
    print("-" * 50)
