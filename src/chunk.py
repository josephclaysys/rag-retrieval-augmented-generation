from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load the PDF
loader = PyPDFLoader("data/document.pdf")
documents = loader.load()

# 2. Create a text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# 3. Split the document
chunks = text_splitter.split_documents(documents)

# 4. Print information
print(f"Number of chunks: {len(chunks)}")

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk.page_content)
