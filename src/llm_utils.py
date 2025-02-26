import os
from langchain_openai import AzureChatOpenAI
from langchain_openai import ChatOpenAI
import pandas as pd

from src.constant import Constants

# import mlflow
# mlflow.set_tracking_uri(Constants.MLFLOW_URI.value)
# mlflow.langchain.autolog()

from src.logger import setup_logger
LOGGER = setup_logger(__name__)

# Environment Variables for Azure OpenAI
os.environ["AZURE_OPENAI_API_KEY"] = os.environ["AZUREOPENAI_API_KEY"]
os.environ["AZURE_OPENAI_ENDPOINT"] = os.environ["AZUREOPENAI_ENDPOINT"]
API_VERSION = os.environ["AZUREOPENAI_API_VERSION"]

# Azure OpenAI Deployment Name
AZUREOPENAI_DEPLOYMENT_NAME = os.environ["AZUREOPENAI_DEPLOYMENT"]

azllm = AzureChatOpenAI(
    azure_deployment=AZUREOPENAI_DEPLOYMENT_NAME,  # or your deployment
    api_version=API_VERSION,  # or your api version
    temperature=0.2
)

# opllm = ChatOpenAI(
#     model="gpt-4o",
#     temperature=0.5
# )

llm = azllm

def generate_response(query, context):

    LOGGER.debug(f"Query received - {query} with context - {context}")
    combined_input = f"Context: {context}\n\nQuery: {query}"
    response = llm(combined_input)
    LOGGER.debug(f"Response generated - {response}")
    return response

def generate_response_for_query(query):

    LOGGER.debug(f"Query received - {query}")
    response_full = llm.invoke(query)
    response = response_full.content
    LOGGER.debug(f"Response generated - {response}")
    return eval(response)

def generate_structured_response(query, context, dm):
    LOGGER.debug(f"Query received - {query} with context - {context}")
    str_llm = llm.with_structured_output(dm)
    combined_input = f"Context: {context}\n\nQuery: {query}"
    response = str_llm.invoke(combined_input)
    LOGGER.debug(f"Response generated - {response}")
    return response


def requirement_func_gaps(query):

    LOGGER.debug("requirement fucn query received")
    gaps_prompt_path = r"src/prompts_template/story_functional_feeback_prompt.txt"

    with open(gaps_prompt_path, "r") as f:
        gaps_prompt = f.read()

    gaps_prompt = gaps_prompt.replace("{STORY_DESC}", query)
    response = llm.invoke(gaps_prompt)
    LOGGER.debug(f"Gaps response: {response}")
    try:
        gaps = eval(response.content)  # Replace eval with json.loads if it's valid JSON
        return True, gaps
    except Exception as e:
        LOGGER.debug(f"Error parsing response: {e}")
        gaps = []
        return False, gaps
    
def requirement_tech_gaps(query):

    LOGGER.debug("requirement tech query received")

    gaps_prompt_path = r"src/prompts_template/tech_requirement_feedback_prompt.txt"

    with open(gaps_prompt_path, "r") as f:
        gaps_prompt = f.read()

    gaps_prompt = gaps_prompt.replace("{STORY_DESC}", query)

    response = llm.invoke(gaps_prompt)
    LOGGER.debug(f"Gaps response: {response}")
    try:
        gaps = eval(response.content)  # Replace eval with json.loads if it's valid JSON
        return True, gaps
    except Exception as e:
        LOGGER.debug(f"Error parsing response: {e}")
        gaps = []
        return False, gaps
    
def get_testing_cases(query):

    LOGGER.debug("testing query received")

    tests_prompt_path = r"src/prompts_template/testing_prompt.txt"

    with open(tests_prompt_path, "r") as f:
        tests_prompt = f.read()

    tests_prompt = tests_prompt.replace("{STORY_DESC}", query)

    response = llm.invoke(tests_prompt)
    LOGGER.debug(f"testing response: {response}")
    try:
        # Remove the ```python block from the response content
        response_content = response.content.strip()
        if response_content.startswith("```python"):
            response_content = response_content[9:]  # Remove the opening ```python
        if response_content.endswith("```"):
            response_content = response_content[:-3]  # Remove the closing ```

        testing_cases = eval(response_content)  # Replace eval with json.loads if it's valid JSON
        return True, testing_cases
    except Exception as e:
        LOGGER.debug(f"Error parsing response: {e}")
        testing_cases = []
        return False, testing_cases
    
def acceptance_criteria(query, testing_cases):


    LOGGER.debug("acceptance query received")

    acceptance_prompt_path = r"src/prompts_template/acceptance_prompt.txt"

    with open(acceptance_prompt_path, "r") as f:
        acceptance_prompt = f.read()

    acceptance_prompt = acceptance_prompt.replace("{STORY_DESC}", query)

    df_test = pd.DataFrame(testing_cases)
    df_test_string = df_test.to_string(index=False)
    acceptance_prompt = acceptance_prompt.replace("{TEST_CRIT}", df_test_string)

    response = llm.invoke(acceptance_prompt)
    LOGGER.debug(f"Gaps response: {response}")
    try:
        accept = eval(response.content)  # Replace eval with json.loads if it's valid JSON
        return True, accept
    except Exception as e:
        LOGGER.debug(f"Error parsing response: {e}")
        gaps = []
        return False, accept

import threading

def call_and_store(key, func, results, *args):
    try:
        flag, result = func(*args)
        results[key] = result if flag else []
    except Exception as e:
        LOGGER.debug(f"Error in {key}: {e}")
        results[key] = []

def gather_results(query):

    LOGGER.debug(f"Query received - {query}")

    results = {}

    threads = [
        threading.Thread(target=call_and_store, args=("func_gaps", requirement_func_gaps, results, query)),
        threading.Thread(target=call_and_store, args=("tech_gaps", requirement_tech_gaps, results, query)),
        threading.Thread(target=call_and_store, args=("testing_cases", get_testing_cases, results, query)),
    ]

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Now we need to process acceptance criteria which depends on testing_cases
    accept_flag, accept = acceptance_criteria(query, results.get("testing_cases", []))
    results["accept"] = accept if accept_flag else []

    return results

def prompt_guardrails(query):

    LOGGER.debug(f"Query received for guardrails - {query}")

    guardrails_prompt_path = r"src/prompts_template/prompt_guardrail.txt"

    with open(guardrails_prompt_path, "r") as f:
        guardrails_prompt = f.read()

    guardrails_prompt = guardrails_prompt.replace("{STORY_DESC}", query)

    response = llm.invoke(guardrails_prompt)
    LOGGER.debug(f"Guardrails response: {response}")

    try:
        guardrails = eval(response.content)  # Replace eval with json.loads if it's valid JSON
        return True, guardrails
    except Exception as e:
        LOGGER.debug(f"Error parsing response: {e}")
        guardrails = {}
        return False, guardrails

