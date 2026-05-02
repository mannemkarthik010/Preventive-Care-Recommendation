import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "chroma_db"
DATA_PATH = "data"

def init_vector_store():
    """Initializes the vector store with documents from the data directory."""
    # 1. Load Documents
    documents = []
    if not os.path.exists(DATA_PATH):
        print(f"Data directory '{DATA_PATH}' not found. Please run data_generation.py first.")
        return None

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            documents.extend(loader.load())

    if not documents:
        print("No PDFs found in the data directory.")
        return None

    # 2. Split Text
    # 512 chunks with 64 overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\\n\\n", "\\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    # 3. Create Embeddings & Store in Chroma
    # Using a fast, local HF model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Check if DB already exists, clear it for a fresh build
    if os.path.exists(CHROMA_PATH):
        import shutil
        shutil.rmtree(CHROMA_PATH)
        
    db = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")
    return db

def get_vector_store():
    """Returns the existing vector store."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    return db

if __name__ == "__main__":
    init_vector_store()
