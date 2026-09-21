# Retrieval-Augmented Generation (RAG) Pipeline

A simple Retrieval-Augmented Generation system that loads information from a PDF, splits the document into chunks, converts the chunks into embeddings, stores them in ChromaDB, retrieves relevant information, and generates grounded answers using a local Qwen language model.

## Overview

This project demonstrates the core components of a Retrieval-Augmented Generation pipeline.

Instead of asking a language model to answer a question entirely from its internal knowledge, the system first retrieves relevant information from a provided document and then uses that information as context for generating the answer.

The pipeline is:

```text
PDF Document
     ↓
Document Loading
     ↓
Text Chunking
     ↓
Embedding Generation
     ↓
ChromaDB Vector Store
     ↓
Similarity Search
     ↓
Relevant Context
     ↓
Qwen 2.5 LLM
     ↓
Grounded Answer
Why RAG?

Large language models can sometimes generate information that is not supported by the provided data.

RAG addresses this by retrieving relevant information from an external knowledge source before generating the answer.

The main advantages include:

Grounding responses in provided documents
Reducing unsupported answers
Using external or private knowledge
Updating knowledge without retraining the language model
Improving transparency by inspecting retrieved context
Project Architecture
                    ┌─────────────────┐
                    │   PDF Document  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  PyPDFLoader    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    Chunking     │
                    │ 500 / overlap50 │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Embeddings    │
                    │ MiniLM-L6-v2    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    ChromaDB     │
                    │  Vector Store   │
                    └────────┬────────┘
                             │
                      Similarity Search
                             │
                             ▼
                    ┌─────────────────┐
                    │ Retrieved       │
                    │ Documents       │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Qwen 2.5:3b     │
                    │ Local LLM       │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    Answer       │
                    └─────────────────┘
Components
1. Document Loading

The project uses LangChain's PyPDFLoader to load the PDF document.

The document is loaded from:

data/document.pdf

The loader extracts the text from each page and represents it as a LangChain document.

2. Text Chunking

The extracted document is divided into smaller chunks using:

RecursiveCharacterTextSplitter

Current configuration:

chunk_size = 500
chunk_overlap = 50

The overlap helps preserve context between neighboring chunks.

3. Embeddings

The project uses:

sentence-transformers/all-MiniLM-L6-v2

to convert text chunks into numerical vector representations.

These embeddings allow semantically similar text to be identified during retrieval.

4. Vector Database

The embeddings are stored using:

ChromaDB

The collection is:

rag_documents

The vector database is persisted locally in:

chroma_db/
5. Retrieval

The system uses similarity search to retrieve the most relevant chunks for a question.

The current configuration retrieves:

k = 3

The retrieval process also exposes the distance associated with each result, which can be used to analyze retrieval quality.

6. Local Language Model

The project uses Ollama with:

qwen2.5:3b

The model runs locally and receives the retrieved document context together with the user's question.

The generation temperature is set to:

0

to encourage deterministic responses.

Grounded Generation

The generation prompt instructs the model to:

Use only the retrieved context.
Avoid outside knowledge.
State when the answer is not present in the context.
Avoid guessing or inventing information.

For example, when the required information is not available in the document, the system is instructed to return:

I don't know based on the provided context.

This provides a basic safeguard against unsupported answers.

Example

Question:

What is machine learning?

The system:

1. Converts the question into an embedding
2. Searches ChromaDB
3. Retrieves the top 3 relevant chunks
4. Builds a context from the retrieved chunks
5. Sends the context and question to Qwen
6. Generates a grounded answer
Retrieval Scores

The retrieval process uses ChromaDB similarity search with scores.

Example output:

--- Result 1 ---
Distance: 0.49

--- Result 2 ---
Distance: 0.86

--- Result 3 ---
Distance: 1.14

The exact interpretation of the distance depends on the vector store and distance metric. In this project, lower distance values indicate more similar results.

These scores can be useful when evaluating retrieval quality and selecting appropriate retrieval thresholds.

Project Structure
rag_project/
│
├── data/
│   └── document.pdf
│
├── chroma_db/
│   └── Local ChromaDB database
│
├── src/
│   ├── chunk.py
│   ├── embed.py
│   ├── generate.py
│   ├── load.py
│   └── retrieve.py
│
├── main.py
├── .gitignore
└── README.md

The local document and ChromaDB database are excluded from Git using .gitignore.

File Responsibilities
src/load.py

Loads the PDF and displays basic document information such as the number of pages and extracted text.

src/chunk.py

Splits the loaded document into smaller overlapping chunks.

src/embed.py

Creates embeddings for the chunks and stores them in ChromaDB.

src/retrieve.py

Performs similarity search against the ChromaDB vector store.

src/generate.py

Combines retrieval with the Qwen language model to generate a grounded answer.

main.py

Entry point for the project.

Installation

Create and activate a Python environment:

conda create -n rag_env python=3.12
conda activate rag_env

Install the required packages:

pip install langchain langchain-community langchain-chroma langchain-huggingface langchain-ollama langchain-text-splitters sentence-transformers chromadb

Ollama is also required for running the local language model.

Pull the Qwen model:

ollama pull qwen2.5:3b
Running the Pipeline
1. Load the Document
python src/load.py
2. Create Chunks
python src/chunk.py
3. Generate Embeddings
python src/embed.py

This creates the local ChromaDB vector store.

4. Test Retrieval
python src/retrieve.py
5. Generate an Answer
python src/generate.py
Technologies
Python
LangChain
ChromaDB
Sentence Transformers
Hugging Face
Ollama
Qwen 2.5
PyPDFLoader
Current Limitations

The current implementation is intentionally simple and is primarily designed for learning and experimentation.

Current limitations include:

Single-document workflow
Fixed chunk size
Fixed top-k retrieval
No retrieval threshold
No reranking
No formal retrieval evaluation
No conversation memory
No web interface
No production deployment
Future Improvements

Potential improvements include:

Retrieval quality evaluation
Similarity thresholding
Dynamic top-k retrieval
Reranking
Hybrid search
Metadata filtering
Query rewriting
Multi-document support
Conversation memory
Retrieval evaluation metrics
Better prompt engineering
Production API
Web-based interface
Learning Objectives

This project is being developed to understand the fundamentals of Retrieval-Augmented Generation, including:

Document ingestion
Chunking strategies
Embeddings
Vector databases
Similarity search
Retrieval quality
Prompt construction
Grounded generation
Hallucination reduction
Local LLM inference
