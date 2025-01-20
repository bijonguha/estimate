import os
from langchain_openai import AzureChatOpenAI
from langchain_openai import ChatOpenAI

# Environment Variables for Azure OpenAI
os.environ["AZURE_OPENAI_API_KEY"] = os.environ["AZUREOPENAI_API_KEY"]
os.environ["AZURE_OPENAI_ENDPOINT"] = os.environ["AZUREOPENAI_ENDPOINT"]
API_VERSION = os.environ["AZUREOPENAI_API_VERSION"]

# Azure OpenAI Deployment Name
AZUREOPENAI_DEPLOYMENT_NAME = os.environ["AZUREOPENAI_DEPLOYMENT"]

azllm = AzureChatOpenAI(
    azure_deployment=AZUREOPENAI_DEPLOYMENT_NAME,  # or your deployment
    api_version=API_VERSION,  # or your api version
    temperature=0.5
)

opllm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.5
)

llm = opllm

def generate_response(query, context):

    combined_input = f"Context: {context}\n\nQuery: {query}"
    response = llm(combined_input)
    return response

def generate_structured_response(query, context, dm):

    str_llm = llm.with_structured_output(dm)
    combined_input = f"Context: {context}\n\nQuery: {query}"
    response = str_llm.invoke(combined_input)
    return response