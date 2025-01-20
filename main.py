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
from src.datamodels import FeatureResponse

LOGGER = setup_logger(__name__)

vectorstores_metadata = load_vectorstore_metadata()

def get_app() -> FastAPI:
    """
    Returns the FastAPI app object
    """
    try:
        fast_app = FastAPI(
                title="Jira Copilot Backend",
                description="A simple FastAPI backend for Mr. Agile application.")
        return fast_app
    except Exception as e:
        LOGGER.error('exception occured in get_app() - {0}'.format(e))

app = get_app()

@app.get("/health", tags=["health"])
async def health_check(request: Request):
    """
    Health check endpoint
    """
    return {"status": 200}


@app.post("/upload_pdf/")
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

@app.post("/query_llm/")
async def query_llm(query: str = Form(...), vectorstore_id: Optional[str] = Form(None)):
    context = ""

    if vectorstore_id:
        if vectorstore_id not in vectorstores_metadata:
            return JSONResponse(content={"error": "Invalid vectorstore_id. Please provide a valid ID."}, status_code=400)

        # Load vectorstore dynamically from disk
        vectorstore_directory = vectorstores_metadata[vectorstore_id]
        client = chromadb.PersistentClient(path=vectorstore_directory)
        collection_name = vectorstore_id
        collection = client.get_collection(name=collection_name)

        relevant_chunks = retrieve_relevant_chunks(query, collection)
        context = "\n".join(relevant_chunks)

    response = generate_response(query, context)
    return JSONResponse(content={"response": response.content}, status_code=200)

@app.post("/generate_subtasks/")
async def generate_subtasks(feature_details: str = Form(...), vectorstore_id: Optional[str] = Form(None)):
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
