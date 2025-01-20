from dotenv import load_dotenv
load_dotenv()

import shutil
import uuid
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile, Form
from typing import Optional

import chromadb

from src.logger import setup_logger
from src.rag_utils import (create_vectorstore, retrieve_relevant_chunks,
                           load_vectorstore_metadata, save_vectorstore_metadata)
from src.llm_utils import generate_response, generate_structured_response
from src.datamodels import FeatureResponse, QuerySimple

LOGGER = setup_logger(__name__)

vectorstores_metadata = load_vectorstore_metadata()

def get_app() -> FastAPI:
    """
    Returns the FastAPI app object
    """
    try:
        fast_app = FastAPI(
                title="EstiMATE",
                description="A simple FastAPI backend for easy estimation of Software Requirements")
        return fast_app
    except Exception as e:
        LOGGER.error('exception occured in get_app() - {0}'.format(e))

app = get_app()

@app.get("/health", tags=["Health"])
async def health_check(request: Request):
    """
    Health check endpoint
    """
    return {"status": 200}


@app.post("/upload_pdf/", tags=["Data Management"])
async def upload_pdf(file: UploadFile = File(...)):
    vectorstore_id = str(uuid.uuid4())
    session_directory = f"src/vec_db/vectorstores/{vectorstore_id}"
    os.makedirs(session_directory, exist_ok=True)
    pdf_path = os.path.join(session_directory, file.filename)

    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    vectorstore = create_vectorstore(pdf_path, session_directory)
    vectorstores_metadata[vectorstore_id] = session_directory
    save_vectorstore_metadata(vectorstores_metadata)

    return JSONResponse(content={"message": "PDF uploaded and vectorstore created successfully.", "vectorstore_id": vectorstore_id}, status_code=200)

@app.post("/query_llm/", tags=["LLM Querying"])
async def query_llm(input_data : QuerySimple):
    
    LOGGER.debug(f"Query received in query_llm - {input_data}")
    
    query = input_data.query
    vectorstore_id = input_data.vectorstore_id

    context = ""

    if vectorstore_id:

        if vectorstore_id not in vectorstores_metadata:
            LOGGER.error(f"Invalid vectorstore_id. Please provide a valid ID.")
            return JSONResponse(content={"error": "Invalid vectorstore_id. Please provide a valid ID."}, status_code=400)

        LOGGER.debug(f"Vectorstore location - {vectorstores_metadata[vectorstore_id]}")

        # Load vectorstore dynamically from disk
        vectorstore_directory = vectorstores_metadata[vectorstore_id]
        client = chromadb.PersistentClient(path=vectorstore_directory)
        collection_name = vectorstore_id
        collection = client.get_collection(name=collection_name)
        LOGGER.debug(f"Chroma client with collection name - {collection_name} loaded")
        relevant_chunks = retrieve_relevant_chunks(query, collection)
        context = "\n".join(relevant_chunks)
        LOGGER.debug(f"Relevant chunks found from vector store - {relevant_chunks}")

    response = generate_response(query, context)
    return JSONResponse(content={"response": response.content}, status_code=200)

@app.post("/generate_subtasks/", tags=["LLM Querying"])
async def generate_subtasks(input_data : QuerySimple):

    feature_details = input_data.query
    vectorstore_id = input_data.get(vectorstore_id, None)

    context = ""

    if vectorstore_id:
        if vectorstore_id not in vectorstores_metadata:
            return JSONResponse(content={"error": "Invalid vectorstore_id. Please provide a valid ID."}, status_code=400)

        # Load vectorstore dynamically from disk
        vectorstore_directory = vectorstores_metadata[vectorstore_id]
        client = chromadb.PersistentClient(path=vectorstore_directory)
        collection_name = vectorstore_id
        collection = client.get_collection(name=collection_name)

        relevant_chunks = retrieve_relevant_chunks(feature_details, collection)
        context = "\n".join(relevant_chunks)

    prompt = (
        f"The product owner has provided the following feature details:\n{feature_details}\n\n"
        f"Context from the system (if any):\n{context}\n\n"
        "Based on the above, generate subtasks for implementation, with acceptance criteria and testing details. Also include important testing scenarios and corner cases to keep in mind for testing the feature"
    )

    response = generate_structured_response(prompt, context, FeatureResponse)

    return JSONResponse(content=response.dict(), status_code=200)
