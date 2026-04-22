from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# 👉 Use raw string for Windows path
PDF_PATH = r"E:\swarupa\swaroopa_dyp_clg\RAG\deep-learning-book.pdf"

print("📘 Loading PDF...")
loader = PyMuPDFLoader(PDF_PATH)
docs = loader.load()

print(f"📄 Total pages loaded: {len(docs)}")

# 👉 Split into chunks (optimized)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)

print(f"🔹 Total chunks created: {len(chunks)}")

# 👉 Embedding model (MUST be same everywhere)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 👉 Store in Chroma DB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma-db"
)

print("✅ PDF successfully ingested into Chroma DB!")