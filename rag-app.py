import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# -------------------------------------------------------
# 🌌 BACKGROUND & STYLING
# -------------------------------------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background:
        linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.92)),
        url("https://images.stockcake.com/public/c/7/d/c7d334f3-aa59-4a9a-b77c-afe2ce609223_large/digital-brain-illuminated-stockcake.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

h1, h2, h3, h4, h5, h6, p, label {
    color: #ffffff !important;
}

/* Question Input Styling - Remove border and keep background white */
[data-testid="stTextInput"] div[data-baseweb="input"] {
    border: none !important;
    background-color: #ffffff !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}

input {
    color: #000000 !important;
}

/* =========================================
   🔘 BUTTON STYLING
   ========================================= */

/* The Ask Button */
.stButton > button {
    background: #000000 !important;
    color: #ffffff !important;
    font-weight: 900 !important;
    border-radius: 10px !important;
    border: 2px solid #333 !important;
    width: 100%;
}

/* The Upload Button - Forced Black for icon and ALL text */
[data-testid="stFileUploader"] button {
    background: #ffffff !important;
    color: #000000 !important;
    font-weight: 900 !important;
    border-radius: 10px !important;
    border: 1px solid #ffffff !important;
}

/* Targets the specific text and "Browse files" label inside the button */
[data-testid="stFileUploader"] button * {
    color: #000000 !important;
    fill: #000000 !important;
}

/* =========================
   💬 CHAT UI (WHITE TEXT)
   ========================= */
.user-msg {
    background: rgba(255,255,255,0.1);
    padding: 12px;
    border-radius: 12px;
    margin: 6px 0;
    text-align: right;
    color: #ffffff !important;
    border: 1px solid #ffffff;
}

.ai-msg {
    background: rgba(255,255,255,0.05);
    padding: 12px;
    border-radius: 12px;
    margin: 6px 0;
    color: #ffffff !important;
    border: 1px solid #ffffff;
}

.title {
    font-size: 42px;
    font-weight: 900;
    text-align: center;
    color: #00eaff;
}

.subtitle {
    text-align: center;
    color: #bdefff;
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# HEADER
# -------------------------------------------------------
st.markdown("<div class='title'>🤖 AI RAG Chatbot</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload PDF → Ask Questions → Get Smart Answers</div>", unsafe_allow_html=True)

# -------------------------------------------------------
# RAG SETUP
# -------------------------------------------------------
uploaded_pdf = st.file_uploader("📄 Upload your PDF here", type=["pdf"])

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if uploaded_pdf:
    if st.session_state.vectorstore is None:
        st.success("PDF uploaded successfully!")
        temp_dir = tempfile.mkdtemp()
        pdf_path = os.path.join(temp_dir, uploaded_pdf.name)

        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.read())

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.vectorstore = Chroma.from_documents(chunks, embedding=embeddings)

# -------------------------------------------------------
# CHAT SECTION
# -------------------------------------------------------
if uploaded_pdf and st.session_state.vectorstore:

    if "chat" not in st.session_state:
        st.session_state.chat = []

    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatMistralAI(model="mistral-small")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Use the provided context to answer. 
        If the context is insufficient, use your knowledge to provide a complete technical answer, 
        prioritizing information found in the document."""),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}")
    ])

    query = st.text_input("💬 Ask your question:")

    if st.button("Ask") and query:
        docs = retriever.invoke(query)
        context = "\n\n".join([d.page_content for d in docs])

        response = llm.invoke(prompt.invoke({
            "context": context,
            "question": query
        }))

        st.session_state.chat.append(("user", query))
        st.session_state.chat.append(("ai", response.content))

    for role, msg in reversed(st.session_state.chat):
        if role == "user":
            st.markdown(f"<div class='user-msg'>{msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='ai-msg'>{msg}</div>", unsafe_allow_html=True)
