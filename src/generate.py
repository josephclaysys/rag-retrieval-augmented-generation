from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

# 1. Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Connect to our existing ChromaDB
vectorstore = Chroma(
    collection_name="rag_documents",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# 3. Create the retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# 4. User question
question = "What is machine learning?"

# 5. Retrieve relevant chunks
results = vectorstore.similarity_search_with_score(
    question,
    k=3
)

documents = [document for document, score in results]

print("\nRetrieved Documents:")

for i, (document, score) in enumerate(results):
    print(f"\n--- Document {i + 1} ---")
    print(f"Distance: {score}")
    print(document.page_content)
print("\nRetrieved Documents:")

for i, document in enumerate(documents):
    print(f"\n--- Document {i + 1} ---")
    print(document.page_content)

# 6. Combine retrieved chunks into context
context = "\n\n".join(
    document.page_content for document in documents
)

# 7. Create the RAG prompt
prompt = f"""
You are a helpful assistant that answers questions using the provided context.

Rules:
1. Use only the information provided in the context.
2. Do not use outside knowledge.
3. If the answer is not present in the context, say:
   "I don't know based on the provided context."
4. Do not make up or guess information.

Context:
{context}

Question:
{question}

Answer:
"""

# 8. Connect to our local Qwen model
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)

# 9. Generate the answer
response = llm.invoke(prompt)

print("\nAnswer:")
print(response.content)
