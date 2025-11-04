from langchain_community.document_loaders import DirectoryLoader
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
#from langchain_community.embeddings import HuggingFaceEmbeddings
from myembedder import MyEmbedder
from langchain_community.vectorstores import Chroma
import shutil
import os



DATA_PATH = "data"
CHROMA_PATH = "chroma"

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents

# Một Document gồm 2 field là page_content (string) và metadata(dictionary)
def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap = 150,
        length_function = len,
        add_start_index = True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)     # rmtree : xóa cây thư mục
    embedder = MyEmbedder()
    print('Create embedder: ', embedder)
    # Create a new DB from the documents]
    db = Chroma.from_documents(
        chunks[:32], embedding=embedder, persist_directory=CHROMA_PATH
    )
    for i in range(32, len(chunks), 32):
        documents = chunks[i:i+32]
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        db.add_texts(texts=texts, metadatas = metadatas)

    db.persist()
    print(f"Saved {db.__len__()} chunks to {CHROMA_PATH}.")


