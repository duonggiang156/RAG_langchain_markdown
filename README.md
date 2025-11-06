This is a RAG (Retrieval-Augmented Generation) system that enables question-answering over custom documents using LLMs. Here's what it does:

  Core Functionality

  1. Document Ingestion (create_database.py:16)
  - Loads markdown documents from the data/ directory
  - Splits documents into chunks (300 chars with 150 char overlap)
  - Generates embeddings using a custom embedding service
  - Stores embeddings in a ChromaDB vector database

  2. Question Answering (query_data.py:27)
  - Takes a text query via command line
  - Searches the vector database for relevant document chunks
  - Uses similarity search to find the top 3 most relevant chunks
  - Sends the retrieved context + question to Google's Gemini 2.5 Pro LLM
  - Returns an answer with source citations

  Key Components

  - Custom Embedder (myembedder.py:7): Calls a remote embedding API at http://101.99.3.94:8080/embed
  - Vector Store: ChromaDB for semantic search
  - LLM: Google Gemini 2.5 Pro for generating answers
  - Framework: LangChain for orchestration

  Typical Usage

  # 1. Build the knowledge base from markdown files
  python main.py

  # 2. Ask questions about your documents
  python query_data.py "your question here"

  The system retrieves relevant context from your documents and generates accurate answers based only on that context, reducing hallucinations.