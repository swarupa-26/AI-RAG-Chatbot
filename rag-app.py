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
# 🌌 DARK AI BACKGROUND
# -------------------------------------------------------
st.markdown("""
<style>

/* =========================
   🌌 BACKGROUND
   ========================= */
[data-testid="stAppViewContainer"] {
    background:
        linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.92)),
        url("https://images.stockcake.com/public/c/7/d/c7d334f3-aa59-4a9a-b77c-afe2ce609223_large/digital-brain-illuminated-stockcake.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* =========================
   💡 TEXT FIX
   ========================= */
h1, h2, h3, h4, h5, h6, p, label {
    color: #ffffff !important;
}

/* =========================
   🔘 ASK BUTTON (BLACK + WHITE TEXT)
   ========================= */
button[kind="primary"] {
    background: #000000 !important;   /* BLACK */
    color: #ffffff !important;        /* WHITE TEXT */
    font-weight: 900 !important;
    border-radius: 10px !important;
    border: 1px solid #333 !important;
}

/* Hover */
button[kind="primary"]:hover {
    background: #111111 !important;
    color: #ffffff !important;
}

/* =========================
   📂 UPLOAD BOX (BLACK + WHITE TEXT)
   ========================= */
.stFileUploader > div {
    background: #000000 !important;
    border: 2px solid #333 !important;
    border-radius: 12px !important;
    padding: 10px !important;
}

/* Upload label */
.stFileUploader label {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* Upload text */
.stFileUploader div {
    color: #ffffff !important;
}

/* Upload button inside */
button[data-testid="stFileUploaderDropzone"] {
    background: #000000 !important;
    color: #ffffff !important;
    border: 1px solid #333 !important;
    font-weight: 800 !important;
}

/* =========================
   INPUT FIX
   ========================= */
input, textarea {
    background: rgba(255,255,255,0.12) !important;
    color: #ffffff !important;
}

input::placeholder {
    color: rgba(255,255,255,0.6) !important;
}

/* =========================
   HEADER
   ========================= */
[data-testid="stHeader"] {
    background: transparent;
}

/* =========================
   TITLE
   ========================= */
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

/* =========================
   CARD UI
   ========================= */
.card {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 18px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.2);
}

/* =========================
   CHAT UI
   ========================= */
.user-msg {
    background: rgba(255,180,0,0.35);
    padding: 12px;
    border-radius: 12px;
    margin: 6px 0;
    text-align: right;
    color: #ffffff !important;
}

.ai-msg {
    background: rgba(0,200,255,0.35);
    padding: 12px;
    border-radius: 12px;
    margin: 6px 0;
    color: #ffffff !important;
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
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    uploaded_pdf = st.file_uploader("📄 Upload your PDF here", type=["pdf"])

    retriever = None
    llm = None
    prompt = None

    if uploaded_pdf:
        st.success("PDF uploaded successfully!")

        temp_dir = tempfile.mkdtemp()
        pdf_path = os.path.join(temp_dir, uploaded_pdf.name)

        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.read())

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = Chroma.from_documents(
            chunks,
            embedding=embeddings
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        llm = ChatMistralAI(model="mistral-small")

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

    for role, msg in st.session_state.chat:
        if role == "user":
            st.markdown(f"<div class='user-msg'>{msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='ai-msg'>{msg}</div>", unsafe_allow_html=True)
