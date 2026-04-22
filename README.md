AI RAG Chatbot

This project implements a complete Retrieval-Augmented Generation (RAG) chatbot pipeline using Mistral LLM, HuggingFace sentence embeddings, and Chroma vector database. It includes:

A PDF ingestion pipeline (createdb.py)
A terminal-based RAG chatbot (main.py)
A full Streamlit UI for uploading PDFs and chatting with them

The system retrieves the most relevant document chunks and generates context-aware answers strictly based on the provided content.

Features
PDF ingestion using PyMuPDF
Text chunking via RecursiveCharacterTextSplitter
Embeddings generated using sentence-transformers/all-MiniLM-L6-v2
Vector storage and search using Chroma
RAG-based generation with Mistral (mistral-small)
Terminal chatbot for quick testing
Fully designed Streamlit chatbot UI with dark gradient + glass effect
Session-based chat history
Strict anti-hallucination prompt

How It Works (RAG Flow)

PDF → Load → Chunk → Embeddings → Chroma DB
                                        ↓
                                  Retriever (k=5/6)
                                        ↓
                               Mistral LLM (RAG Prompt)
                                        ↓
                               Final AI Answer

Key Technologies
LangChain
Mistral AI API
HuggingFace Embeddings
Chroma Vector Store
Streamlit
PyMuPDF
python-dotenv 
