import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Set API Key
os.environ["OPENAI_API_KEY"] = "your_api_key_here"

# 1. Load Documents
loader = PyPDFLoader("data/notes.pdf")
documents = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = splitter.split_documents(documents)

# 3. Convert to embeddings + store in FAISS
embeddings = OpenAIEmbeddings()
vector_db = FAISS.from_documents(docs, embeddings)

# 4. Create Retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# 5. LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# 6. RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 7. Chat loop
print("📚 RAG Study Buddy Ready! Ask your doubts.\n")

while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break

    result = qa_chain(query)

    print("\n🤖 Answer:", result["result"])
    print("\n📄 Sources:")
    for doc in result["source_documents"]:
        print("-", doc.metadata.get("source", "Unknown"))
    print("\n" + "-"*50)
