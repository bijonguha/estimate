import gradio as gr
import requests
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

# FastAPI base URL (assuming the FastAPI server is running locally)
FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000")

def format_response(response):
    """
    Takes the response from FastAPI and returns Gradio components to display it beautifully.
    """
    # Extract data from the response
    testing_cases = response.get('testing_cases', [])
    func_gaps = response.get('func_gaps', [])
    tech_gaps = response.get('tech_gaps', [])
    accept = response.get('accept', [])

    # Convert data to pandas DataFrames
    testing_cases_df = pd.DataFrame(testing_cases)
    func_gaps_df = pd.DataFrame(func_gaps)
    tech_gaps_df = pd.DataFrame(tech_gaps)
    accept_df = pd.DataFrame(accept)

    return func_gaps_df, tech_gaps_df, testing_cases_df, accept_df

def upload_pdf(file):
    """
    Uploads a PDF to FastAPI and returns the vectorstore ID.
    """
    # Ensure that the file is correctly passed as a file object.
    with open(file, 'rb') as f:  # open the file in binary read mode
        files = {'file': (file, f)}  # 'file' should match the form field name in FastAPI
        
        try:
            response = requests.post(f"{FASTAPI_BASE_URL}/upload_pdf/", files=files)
            if response.status_code == 200:
                return response.json().get("vectorstore_id")
            else:
                return f"Error: {response.json().get('error')}"
        except Exception as e:
            return f"Error occurred while uploading: {str(e)}"

def query_llm(query, vectorstore_id=None):
    """
    Queries the LLM with a specific query and vectorstore ID.
    """
    data = {
        "query": query,
        "vectorstore_id": vectorstore_id
    }
    response = requests.post(f"{FASTAPI_BASE_URL}/query_llm/", json=data)
    if response.status_code == 200:
        return response.json().get("response")
    else:
        return f"Error: {response.json().get('error')}"

def generate_subtasks(feature_details, vectorstore_id=None):
    """
    Generates subtasks for the given feature details and vectorstore ID.
    """
    data = {
        "query": feature_details,
        "vectorstore_id": vectorstore_id
    }
    response = requests.post(f"{FASTAPI_BASE_URL}/generate_subtasks/", json=data)
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.json().get('error')}"

def estimate_quality(feature_details, vectorstore_id=None):
    """
    Estimates the quality of a feature based on the details and vectorstore ID.
    """

    data = {
        "query": feature_details,
        "vectorstore_id": vectorstore_id
    }
    response = requests.post(f"{FASTAPI_BASE_URL}/estimate_quality/", json=data)

    if response.status_code == 200:
        formatted_response = format_response(response.json())
        return formatted_response
    else:
        return f"Error: {response.json().get('error')}"

# Gradio UI
with gr.Blocks() as demo:
    with gr.Tab("Upload PDF"):
        pdf_file = gr.File(label="Upload PDF")
        vectorstore_id_output = gr.Textbox(label="Vectorstore ID")
        pdf_file.change(upload_pdf, pdf_file, vectorstore_id_output)  # Use 'change' instead of 'submit'

    # with gr.Tab("Query LLM"):
    #     query_input = gr.Textbox(label="Enter your query")
    #     vectorstore_id_input = gr.Textbox(label="Enter your Vectorstore ID (Optional)")  # No 'optional' argument
    #     llm_response_output = gr.Textbox(label="LLM Response")
    #     query_button = gr.Button("Query LLM")
    #     query_button.click(query_llm, [query_input, vectorstore_id_input], llm_response_output)

    # with gr.Tab("Generate Subtasks"):
    #     feature_details_input = gr.Textbox(label="Feature Details")
    #     vectorstore_id_input = gr.Textbox(label="Enter your Vectorstore ID (Optional)")
    #     generate_subtasks_button = gr.Button("Generate Subtasks")
    #     generate_subtasks_output = gr.JSON(label="Generated Subtasks")
    #     generate_subtasks_button.click(generate_subtasks, [feature_details_input, vectorstore_id_input], generate_subtasks_output)

    with gr.Tab("Estimate Quality"):
        estimate_quality_input = gr.Textbox(label="Feature Details")
        vectorstore_id_input = gr.Textbox(label="Enter your Vectorstore ID (Optional)")
        estimate_quality_button = gr.Button("Estimate Quality")
        func_gaps_output = gr.Dataframe(label="Functional Gaps")
        tech_gaps_output = gr.Dataframe(label="Technical Gaps")
        testing_cases_output = gr.Dataframe(label="Testing Cases")
        accept_output = gr.Dataframe(label="Acceptance Criteria")
        
        # When clicking the button, get the response and show it in respective tables
        estimate_quality_button.click(
            estimate_quality, 
            [estimate_quality_input, vectorstore_id_input],
            [func_gaps_output, tech_gaps_output, testing_cases_output, accept_output]
        )

# Launch Gradio app
demo.launch()
