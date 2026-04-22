from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# SAME embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load Chroma DB
vectorstore = Chroma(
    persist_directory="chroma-db",
    embedding_function=embedding_model
)

# Retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)

# LLM
llm = ChatMistralAI(model="mistral-small")

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """You are a helpful RAG assistant.
Use ONLY the provided context to answer.

If answer is not found in the context,
respond with: "I could not find the answer in the document."
"""),
        ("human",
         """Context:
{context}

Question:
{question}
""")
    ]
)

print("RAG Chatbot Loaded Successfully!")
print("Press 0 to exit.\n")

while True:
    query = input("You: ")

    if query == "0":
        break

    #  Retrieve documents
    docs = retriever.invoke(query)

    if not docs:
        print("\nAI: I could not find the answer in the document.\n")
        continue

    #  Debug
    print("\n Retrieved Chunks:\n")
    for doc in docs:
        print(doc.page_content[:150])

    # Build context
    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    #  Get response
    response = llm.invoke(final_prompt)

    print("\nAI:", response.content, "\n")