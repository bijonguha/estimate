import json
import os
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.logger import setup_logger
from src.constant import Constants

LOGGER = setup_logger(__name__)

vectorstore_metadata_file = Constants.VECSTORE_PATH.value

# Load vectorstore metadata
def load_vectorstore_metadata():

    """Load existing vectorstore metadata from a file."""
    try:
        if os.path.exists(vectorstore_metadata_file):
            with open(vectorstore_metadata_file, "r") as f:
                return json.load(f)
    except Exception as e:
        LOGGER.error(f"Failed to load vectorstore metadata: {e}")
    return {}

# Save vectorstore metadata
def save_vectorstore_metadata(metadata):
    """Save vectorstore metadata to a file."""
    try:
        # Ensure the directory for the metadata file exists
        directory = os.path.dirname(vectorstore_metadata_file)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            LOGGER.info(f"Created directory for vectorstore metadata: {directory}")

        # Save metadata to the file
        with open(vectorstore_metadata_file, "w") as f:
            json.dump(metadata, f)
        LOGGER.info("Vectorstore metadata saved successfully.")
    except Exception as e:
        LOGGER.error(f"Failed to save vectorstore metadata: {e}")

# Create vectorstore
def create_vectorstore(pdf_path: str, persist_directory: str):
    """Load a PDF, split it into chunks, and create a vectorstore."""
    try:
        LOGGER.info(f"Processing PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path=persist_directory)
        collection_name = os.path.basename(persist_directory)  # Use directory name as collection name
        collection = client.get_or_create_collection(name=collection_name)

        # Prepare documents and metadata
        documents_content = [chunk.page_content for chunk in chunks]
        metadata = [{"page": idx, "content": chunk.page_content} for idx, chunk in enumerate(chunks)]
        ids = [f"chunk_{idx}" for idx in range(len(chunks))]

        # Add documents to the collection
        collection.add(documents=documents_content, metadatas=metadata, ids=ids)

        LOGGER.info("Vectorstore created and persisted successfully.")
        return client
    except Exception as e:
        LOGGER.error(f"Failed to create vectorstore: {e}")
        raise

# Retrieve relevant chunks
def retrieve_relevant_chunks(query, chroma_collection, top_k=5):
    """Retrieve top-k relevant chunks for a given query from the vectorstore."""
    try:
        LOGGER.info("Retrieving relevant chunks...")
        results = chroma_collection.query(query_texts=[query], n_results=top_k)

        if results and "documents" in results:
            LOGGER.info(f"Retrieved {len(results['documents'][0])} relevant chunks.")
            return results["documents"][0]  # Return the list of top-k documents
        else:
            LOGGER.warning("No relevant chunks found.")
            return []
    except Exception as e:
        LOGGER.error(f"Failed to retrieve relevant chunks: {e}")
        raise
