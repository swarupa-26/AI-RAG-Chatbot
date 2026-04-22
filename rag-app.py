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
# 🌌 UI STYLING (CYBERPUNK THEME)
# -------------------------------------------------------
st.set_page_config(page_title="AI RAG Chatbot", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.92)),
                url("https://images.stockcake.com/public/c/7/d/c7d334f3-aa59-4a9a-b77c-afe2ce609223_large/digital-brain-illuminated-stockcake.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
h1, h2, h3, h4, h5, h6, p, label { color: #ffffff !important; }
.stButton > button {
    background: #000000 !important;
    color: #00eaff !important;
    font-weight: 900 !important;
    border: 2px solid #00eaff !important;
    border-radius: 10px !important;
    width: 100%;
}
.user-msg { background: rgba(255,180,0,0.2); padding: 12px; border-radius: 12px; margin: 6px 0; text-align: right; border: 1px solid #ffb400; }
.ai-msg { background: rgba(0,234,255,0.15); padding: 12px; border-radius: 12px; margin: 6px 0; border: 1px solid #00eaff; }
.title { font-size: 42px; font-weight: 900; text-align: center; color: #00eaff; text-shadow: 0px 0px 10px #00eaff; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🤖 NEURAL RAG EXPLORER</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Advanced Deep Learning Document Assistant</p>", unsafe_allow_html=True)

# -------------------------------------------------------
# RAG CORE LOGIC
# -------------------------------------------------------
uploaded_pdf = st.file_uploader("📄 Upload Deep Learning PDF", type=["pdf"])

# Initialize session state for the vectorstore so it doesn't reload every click
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if uploaded_pdf:
    if st.session_state.vectorstore is None:
        with st.status("🧠 Processing Document into Neural Layers...") as status:
            temp_dir = tempfile.mkdtemp()
            pdf_path = os.path.join(temp_dir, uploaded_pdf.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_pdf.read())

            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
            chunks = splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
            status.update(label="✅ Ingestion Complete!", state="complete")

# -------------------------------------------------------
# CHAT INTERFACE
# -------------------------------------------------------
if st.session_state.vectorstore:
    # 1. Setup Retriever & LLM
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatMistralAI(model="mistral-small")

    # 2. THE IMPROVED PROMPT (The "Brain" Fix)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a specialized Deep Learning Assistant. 
        Your goal is to explain concepts using the provided PDF Context as your primary source.
        
        RULES:
        - If the context contains the answer, summarize it clearly.
        - If the context is missing details or the user uses acronyms (like CNN, RNN, ReLU), use your internal knowledge to provide a full technical explanation.
        - Always mention if a specific detail comes from the document or general deep learning principles.
        - Be technical, precise, and helpful.
        """),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}")
    ])

    if "chat" not in st.session_state:
        st.session_state.chat = []

    query = st.text_input("💬 Query the Neural Network (e.g., 'What is CNN?'):")

    if st.button("EXECUTE SEARCH") and query:
        # Retrieve context
        docs = retriever.invoke(query)
        context_text = "\n\n".join([d.page_content for d in docs])

        # Generate response
        chain_input = {"context": context_text, "question": query}
        response = llm.invoke(prompt.invoke(chain_input))

        st.session_state.chat.append(("user", query))
        st.session_state.chat.append(("ai", response.content))

    # Display Chat
    for role, msg in reversed(st.session_state.chat):
        if role == "user":
            st.markdown(f"<div class='user-msg'><b>You:</b> {msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='ai-msg'><b>AI:</b> {msg}</div>", unsafe_allow_html=True)
else:
    st.info("Please upload a PDF to begin the session.")
