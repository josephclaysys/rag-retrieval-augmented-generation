from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Load the same embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Connect to our existing ChromaDB
vectorstore = Chroma(
    collection_name="rag_documents",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# 3. User question
query = "What is machine learning?"

# 4. Perform similarity search
results = vectorstore.similarity_search_with_score(
    query,
    k=3
)

# 5. Display results
for i, (document, score) in enumerate(results):
    print(f"\n--- Result {i + 1} ---")
    print(f"Score: {score}")
    print(f"Content:\n{document.page_content}")
