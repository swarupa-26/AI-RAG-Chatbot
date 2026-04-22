from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader   # ✅ FIXED
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

load_dotenv()

# ✅ FIX 1: Use relative path
PDF_PATH = "data/deep-learning-book.pdf"

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"PDF not found at: {PDF_PATH}")

print("📘 Loading PDF...")
loader = PyPDFLoader(PDF_PATH)   # ✅ FIXED loader
docs = loader.load()

print(f"📄 Total pages loaded: {len(docs)}")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)

print(f"Total chunks created: {len(chunks)}")

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Store in Chroma DB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma-db"
)

print("✅ PDF successfully ingested into Chroma DB")
