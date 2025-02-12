from dotenv import load_dotenv
load_dotenv()

import shutil
import uuid
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import chromadb

from langchain.prompts.chat import ChatPromptTemplate

from src.logger import setup_logger

from src.rag_utils import (create_vectorstore, retrieve_relevant_chunks,
                           load_vectorstore_metadata, save_vectorstore_metadata)

from src.llm_utils import generate_response, generate_structured_response, gather_results, generate_response_for_query

from src.datamodels import FeatureResponse, QuerySimple, ResponseFile, QueryFeature

LOGGER = setup_logger(__name__)

vectorstores_metadata = load_vectorstore_metadata()

def get_app() -> FastAPI:
    """
    Returns the FastAPI app object
    """
    try:
        fast_app = FastAPI(
                title="EstiMATE",
                description="A simple FastAPI backend for easy estimation of Software Requirements",
                root_path="/api/"
                )
        return fast_app
    except Exception as e:
        LOGGER.error('exception occured in get_app() - {0}'.format(e))

app = get_app()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific domains for better security
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

@app.get("/health", tags=["Health"])
async def health_check(request: Request):
    """
    Health check endpoint
    """
    return {"status": 200}


@app.post("/upload_pdf/", tags=["Data Management"],
          response_model=ResponseFile)
async def upload_pdf(file: UploadFile = File(...)):

    if file.filename in vectorstores_metadata['filenames_dict'].keys():
        LOGGER.debug(f"File already exists in vectorstore - {file.filename}")
        resp = {"vectorstore_id":vectorstores_metadata['filenames_dict'][file.filename]}
        return resp
    
    vectorstore_id = str(uuid.uuid4())
    session_directory = f"vec_db/vectorstores/{vectorstore_id}"
    os.makedirs(session_directory, exist_ok=True)
    pdf_path = os.path.join(session_directory, file.filename)

    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    create_vectorstore(pdf_path, session_directory)
    LOGGER.debug(f"Vector store created for file : {file.filename}")
    
    vectorstores_metadata['filenames_dict'][file.filename] = vectorstore_id
    vectorstores_metadata[vectorstore_id] = session_directory
    save_vectorstore_metadata(vectorstores_metadata)

    resp = {"vectorstore_id":vectorstore_id}
    LOGGER.debug(f"File generated response - {resp}")
    return resp

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
    vectorstore_id = input_data.vectorstore_id

    context = ""

    if vectorstore_id == None:
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

@app.post("/refine_requirements/", tags=["LLM Querying"])
async def refine_requirement(input_data : QueryFeature):

    LOGGER.debug(f"Query received in refine_requirement - {input_data}")

    feature_details = input_data.query
    vectorstore_id = input_data.vectorstore_id

    context = ""

    if vectorstore_id not in vectorstores_metadata:
        return JSONResponse(content={"error": "Invalid vectorstore_id. Please provide a valid ID."}, status_code=400)

    # Load vectorstore dynamically from disk
    vectorstore_directory = vectorstores_metadata[vectorstore_id]
    client = chromadb.PersistentClient(path=vectorstore_directory)
    collection_name = vectorstore_id
    collection = client.get_collection(name=collection_name)

    relevant_chunks = retrieve_relevant_chunks(feature_details, collection)
    context = "\n".join(relevant_chunks)

    story_ref_tmp_path = r"src/prompts_template/story_ref.txt"

    with open(story_ref_tmp_path, "r") as f:
        story_ref_prompt = f.read()

    story_ref_prompt = ChatPromptTemplate.from_template(story_ref_prompt)
    story_ref_message = story_ref_prompt.invoke({"FEAT_REQ":feature_details, "ADD_CONTX":context})

    response = generate_response_for_query(story_ref_message)

    return JSONResponse(content=response, status_code=200)

        
@app.post("/estimate_quality/", tags=["LLM Querying"])
async def estimate_quality(input_data : QuerySimple):

    LOGGER.debug(f"Query received in estimate_quality - {input_data}")
    feature_details = input_data.query
    vectorstore_id = input_data.vectorstore_id

    context = ""

    if vectorstore_id != None and vectorstore_id != "":
        if vectorstore_id not in vectorstores_metadata:
            return JSONResponse(content={"error": "Invalid vectorstore_id. Please provide a valid ID."}, status_code=400)

        LOGGER.debug(f"Vectorstore location - {vectorstores_metadata[vectorstore_id]}")
        # Load vectorstore dynamically from disk
        vectorstore_directory = vectorstores_metadata[vectorstore_id]
        client = chromadb.PersistentClient(path=vectorstore_directory)
        collection_name = vectorstore_id
        collection = client.get_collection(name=collection_name)

        relevant_chunks = retrieve_relevant_chunks(feature_details, collection)
        context = "\n".join(relevant_chunks)

        additional_context_path = r"src/prompts_template/context_doc.txt"

        with open(additional_context_path, "r") as f:
            additional_context_prompt = f.read()

        additional_context = additional_context_prompt.replace("{ADDITIONAL_CONTEXT}", context)

        feature_details = additional_context.replace("{STORY_DESC}", feature_details)
        LOGGER.debug(f"Additional context injected, New query - {feature_details}")

    response = gather_results(feature_details)

    LOGGER.info(f"Final response - {response}")

    return JSONResponse(content=response, status_code=200)
