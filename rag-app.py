import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# FIXED: Use PyPDFLoader (works on Streamlit Cloud)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# -------------------------------------------------------
# 🌗 AUTO DARK/LIGHT MODE ADAPTIVE UI
# -------------------------------------------------------
st.markdown("""
<style>

/* ======================
   SYSTEM DARK MODE
   ====================== */
@media (prefers-color-scheme: dark) {

    /* Background */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 20% 20%, #0f2027, #203a43, #000000);
        color: white !important;
    }

    body, p, label, span, div {
        color: white !important;
    }

    input, textarea {
        background: rgba(255,255,255,0.15) !important;
        color: white !important;
        border: 1px solid white !important;
    }
}

/* ======================
   SYSTEM LIGHT MODE
   ====================== */
@media (prefers-color-scheme: light) {

    /* Light background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(to bottom right, #ffffff, #e7f4ff);
        color: black !important;
    }

    body, p, label, span, div {
        color: black !important;
    }

    input, textarea {
        background: white !important;
        color: black !important;
        border: 1px solid #444 !important;
    }
}

/* ---------- Common UI Styles (Work in both themes) ---------- */

[data-testid="stHeader"] {
    background: transparent;
}

.title {
    font-size: 42px;
    font-weight: 900;
    text-align: center;
    color: #00eaff;
}

.subtitle {
    text-align: center;
    color: #59c8ff;
    margin-bottom: 30px;
}

.card {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 18px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.2);
}

input {
    border-radius: 10px !important;
    padding: 10px !important;
}

/* Button */
button[kind="primary"] {
    background: linear-gradient(135deg, #00eaff, #007bff) !important;
    color: black !important;
    border-radius: 10px !important;
    font-weight: bold !important;
}

/* Chat bubbles */
.user-msg {
    background: rgba(255,180,0,0.35);
    padding: 12px;
    border-radius: 12px;
    margin: 6px 0;
    text-align: right;
}

.ai-msg {
    background: rgba(0,200,255,0.35);
    padding: 12px;
    border-radius: 12px;
    margin: 6px 0;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# HEADER UI
# -------------------------------------------------------
st.markdown("<div class='title'>🤖 AI RAG Chatbot</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload PDF → Ask Questions → Get Smart Answers</div>", unsafe_allow_html=True)

# -------------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------------
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    uploaded_pdf = st.file_uploader("📄 Upload your PDF", type=["pdf"])

    if uploaded_pdf:
        st.success("PDF uploaded!")

        temp_dir = tempfile.mkdtemp()
        pdf_path = os.path.join(temp_dir, uploaded_pdf.name)

        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.read())

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        llm = ChatMistralAI(model="mistral-small")

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Use only the context. If not found, say 'Not found in document.'"),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}")
        ])

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------
# CHAT INTERFACE
# -------------------------------------------------------
if uploaded_pdf:

    if "chat" not in st.session_state:
        st.session_state.chat = []

    query = st.text_input("💬 Ask your question:")

    if st.button("Ask"):
        docs = retriever.invoke(query)

        context = "\n\n".join([d.page_content for d in docs])

        res = llm.invoke(prompt.invoke({
            "context": context,
            "question": query
        }))

        st.session_state.chat.append(("user", query))
        st.session_state.chat.append(("ai", res.content))

    # Display chat bubbles
    for role, msg in st.session_state.chat:
        if role == "user":
            st.markdown(f"<div class='user-msg'>{msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='ai-msg'>{msg}</div>", unsafe_allow_html=True)
