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
# 🌌 FORCE DARK MODE UI + AI BACKGROUND
# -------------------------------------------------------
st.markdown("""
<style>

/* =========================
   🌌 GLOBAL DARK BACKGROUND
   ========================= */

[data-testid="stAppViewContainer"] {
    background:
        linear-gradient(rgba(0,0,0,0.78), rgba(0,0,0,0.88)),
        url("https://images.stockcake.com/public/c/7/d/c7d334f3-aa59-4a9a-b77c-afe2ce609223_large/digital-brain-illuminated-stockcake.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: white !important;
}

/* Force all text visible */
html, body, p, label, span, div {
    color: white !important;
}

/* Inputs */
input, textarea {
    background: rgba(255,255,255,0.12) !important;
    color: white !important;
    border: 1px solid white !important;
}

/* Header */
[data-testid="stHeader"] {
    background: transparent;
}

/* Title */
.title {
    font-size: 42px;
    font-weight: 900;
    text-align: center;
    color: #00eaff;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #59c8ff;
    margin-bottom: 30px;
}

/* Glass card */
.card {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 18px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.2);
}

/* Button */
button[kind="primary"] {
    background: linear-gradient(135deg, #00eaff, #007bff) !important;
    color: black !important;
    border-radius: 10px !important;
    font-weight: bold !important;
}

/* User chat */
.user-msg {
    background: rgba(255,180,0,0.35);
    padding: 12px;
    border-radius: 12px;
    margin: 6px 0;
    text-align: right;
}

/* AI chat */
.ai-msg {
    background: rgba(0,200,255,0.35);
    padding: 12px;
    border-radius: 12px;
    margin: 6px 0;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# HEADER
# -------------------------------------------------------
st.markdown("<div class='title'>🤖 AI RAG Chatbot</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload PDF → Ask Questions → Get Smart Answers</div>", unsafe_allow_html=True)

# -------------------------------------------------------
# FILE UPLOAD + RAG SETUP
# -------------------------------------------------------
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    uploaded_pdf = st.file_uploader("📄 Upload your PDF", type=["pdf"])

    retriever = None
    llm = None
    prompt = None

    if uploaded_pdf:
        st.success("PDF uploaded successfully!")

        temp_dir = tempfile.mkdtemp()
        pdf_path = os.path.join(temp_dir, uploaded_pdf.name)

        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.read())

        # Load PDF
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        # Split text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Vector DB
        vectorstore = Chroma.from_documents(
            chunks,
            embedding=embeddings
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        # LLM
        llm = ChatMistralAI(model="mistral-small")

        # Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Use only the context. If not found, say 'Not found in document.'"),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}")
        ])

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------
# CHAT SECTION
# -------------------------------------------------------
if uploaded_pdf and retriever:

    if "chat" not in st.session_state:
        st.session_state.chat = []

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

    # Display chat history
    for role, msg in st.session_state.chat:
        if role == "user":
            st.markdown(f"<div class='user-msg'>{msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='ai-msg'>{msg}</div>", unsafe_allow_html=True)
