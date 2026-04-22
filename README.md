
This application is a Retrieval-Augmented Generation (RAG) based AI chatbot designed to let users interact with PDF documents and extract accurate, context-aware answers using an LLM.

Live Link of Project  : https://ai-rag-doc-chatbot.streamlit.app/


Overview : 
The app combines document retrieval + AI generation to ensure that answers are not guessed but are grounded in the actual content of the uploaded or preprocessed documents.

It has two modes of interaction:

A terminal-based chatbot for direct querying
A web-based Streamlit interface for an interactive experience


Core Functionality
1. Document Processing
The app reads a PDF using PyMuPDF and splits it into smaller chunks using a text splitter. This ensures that large documents can be efficiently searched.

2. Embedding Generation
Each chunk is converted into a numerical vector using a HuggingFace embedding model (all-MiniLM-L6-v2). These embeddings capture semantic meaning.

3. Vector Storage
The embeddings are stored in a Chroma vector database:
Persistent storage in createdb.py
Temporary in-memory storage in the Streamlit app

4. Retrieval Mechanism
When a user asks a question:
The query is converted into an embedding
The system retrieves the most relevant chunks (top-k results)

5. AI Response Generation
The retrieved context is passed to the Mistral LLM with a strict prompt:

The model must answer only from the given context
If no answer is found, it explicitly says so

Components

Terminal Chatbot (main.py)
Loads prebuilt vector database
Retrieves top 6 relevant chunks
Displays retrieved text (for debugging)
Generates answers using Mistral
Runs in a continuous loop until exit


Database Builder (createdb.py)
Loads a fixed PDF file
Splits it into chunks (800 size, 200 overlap)
Generates embeddings
Stores them in a persistent Chroma DB


Streamlit Web App
Allows users to upload any PDF dynamically
Creates embeddings on the fly
Provides a modern UI with:
Gradient dark theme
Glassmorphism cards
Chat-style interaction
Maintains chat history using session state


Key Characteristics
Context-aware: Answers are based strictly on document content
Low hallucination: Controlled prompt prevents fabricated answers
Modular design: Separate ingestion, retrieval, and UI layers
Scalable: Can be extended to multiple documents or APIs
User-friendly: Both CLI and GUI support

Typical Workflow
Load or upload a PDF
Convert it into chunks
Generate embeddings
Store in vector database
User asks a question
Retrieve relevant chunks
LLM generates answer using context


Use Cases
Studying large textbooks
Document-based Q&A systems
Internal knowledge assistants
Research paper exploration
Technical documentation querying
