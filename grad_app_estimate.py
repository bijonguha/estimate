from dotenv import load_dotenv
load_dotenv()

import os
import gradio as gr

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from markdownify import markdownify

import pandas as pd
import tempfile

endpoint = "https://openai-aiattack-msa-001605-switzerlandnorth-languagegen-00.openai.azure.com"
deployment = "gpt-4-turbo-2024-04-09" 
api_key = "2e4132e05915415cbe6a826be49594ed"
os.environ["AZURE_OPENAI_API_KEY"] = api_key 

azllm = AzureChatOpenAI(
    azure_deployment=deployment,  # or your deployment
    api_version="2023-06-01-preview",  # or your api version
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# opllm = ChatOpenAI(
#     model="gpt-4o",
#     temperature=0.5
# )

llm = azllm

def get_testing_cases(query):

    print("testing query received")

    tests_prompt_path = r"src/prompts_template/testing_prompt.txt"

    with open(tests_prompt_path, "r") as f:
        tests_prompt = f.read()

    tests_prompt = tests_prompt.replace("{STORY_DESC}", query)

    response = llm.invoke(tests_prompt)
    print(f"testing response: {response}")
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
        print(f"Error parsing response: {e}")
        testing_cases = []
        return False, testing_cases

def requirement_func_gaps(query):


    print("requirement fucn query received")

    gaps_prompt_path = r"src/prompts_template/story_functional_feeback_prompt.txt"

    with open(gaps_prompt_path, "r") as f:
        gaps_prompt = f.read()

    gaps_prompt = gaps_prompt.replace("{STORY_DESC}", query)

    response = llm.invoke(gaps_prompt)
    print(f"Gaps response: {response}")
    try:
        gaps = eval(response.content)  # Replace eval with json.loads if it's valid JSON
        return True, gaps
    except Exception as e:
        print(f"Error parsing response: {e}")
        gaps = []
        return False, gaps

def requirement_tech_gaps(query):

    print("requirement tech query received")

    gaps_prompt_path = r"src/prompts_template/tech_requirement_feedback_prompt.txt"

    with open(gaps_prompt_path, "r") as f:
        gaps_prompt = f.read()

    gaps_prompt = gaps_prompt.replace("{STORY_DESC}", query)

    response = llm.invoke(gaps_prompt)
    print(f"Gaps response: {response}")
    try:
        gaps = eval(response.content)  # Replace eval with json.loads if it's valid JSON
        return True, gaps
    except Exception as e:
        print(f"Error parsing response: {e}")
        gaps = []
        return False, gaps

def acceptance_criteria(query, testing_cases):


    print("acceptance query received")

    acceptance_prompt_path = r"src/prompts_template/acceptance_prompt.txt"

    with open(acceptance_prompt_path, "r") as f:
        acceptance_prompt = f.read()

    acceptance_prompt = acceptance_prompt.replace("{STORY_DESC}", query)

    df_test = pd.DataFrame(testing_cases)
    df_test_string = df_test.to_string(index=False)
    acceptance_prompt = acceptance_prompt.replace("{TEST_CRIT}", df_test_string)

    response = llm.invoke(acceptance_prompt)
    print(f"Gaps response: {response}")
    try:
        accept = eval(response.content)  # Replace eval with json.loads if it's valid JSON
        return True, accept
    except Exception as e:
        print(f"Error parsing response: {e}")
        gaps = []
        return False, accept

def generate_csv_report(query, gaps_func, tech_gaps, testing_cases, accept):
    # Define the local file path to save the CSV
    local_file_path = "output_report.csv"

    with open(local_file_path, "w", newline="") as file:
        # Write the query as a header
        file.write(f"User Query:\n{query}\n\n")

        # Write Gaps
        file.write("Gaps in Requirement:\n")
        pd.DataFrame(gaps_func).to_csv(file, index=False)
        file.write("\n")

        file.write("Gaps in Technical Requirements:\n")
        pd.DataFrame(tech_gaps).to_csv(file, index=False)
        file.write("\n")

        # Write Testing Cases
        file.write("Testing Criterias:\n")
        pd.DataFrame(testing_cases).to_csv(file, index=False)
        file.write("\n")

        # Write Acceptance Criteria
        file.write("Acceptance Criteria:\n")
        pd.DataFrame(accept).to_csv(file, index=False)

    print(f"CSV report saved at: {os.path.abspath(local_file_path)}")
    return local_file_path

# def process_input(query):
#     test_flag, testing_cases = get_testing_cases(query)
#     gaps_func_flag, func_gaps = requirement_func_gaps(query)
#     gaps_tech_flag, tech_gaps = requirement_tech_gaps(query)
#     accept_flag, accept = acceptance_criteria(query, testing_cases)

#     csv_path = generate_csv_report(query, func_gaps, tech_gaps, testing_cases, accept)

#     formatted_data = pd.DataFrame(testing_cases)
#     func_gaps_data = pd.DataFrame(func_gaps)
#     tech_gaps_data = pd.DataFrame(tech_gaps)
#     accept_data = pd.DataFrame(accept)

#     print("Dataframes created and CSV generated.")
#     return formatted_data, gr.update(visible=True), func_gaps_data, gr.update(visible=True), tech_gaps_data, gr.update(visible=True), accept_data, gr.update(visible=True), csv_path

def process_input(query):
    # Initialize the dataframes to hold the results for all tables
    formatted_data = pd.DataFrame()
    func_gaps_data = pd.DataFrame()
    tech_gaps_data = pd.DataFrame()
    accept_data = pd.DataFrame()
    csv_path = ""

    # Testing cases
    test_flag, testing_cases = get_testing_cases(query)
    formatted_data = pd.DataFrame(testing_cases)
    yield formatted_data, gr.update(visible=True), func_gaps_data, gr.update(visible=False), tech_gaps_data, gr.update(visible=False), accept_data, gr.update(visible=False), csv_path

    # Functional gaps
    gaps_func_flag, func_gaps = requirement_func_gaps(query)
    func_gaps_data = pd.DataFrame(func_gaps)
    yield formatted_data, gr.update(visible=True), func_gaps_data, gr.update(visible=True), tech_gaps_data, gr.update(visible=False), accept_data, gr.update(visible=False), csv_path

    # Technical gaps
    gaps_tech_flag, tech_gaps = requirement_tech_gaps(query)
    tech_gaps_data = pd.DataFrame(tech_gaps)
    yield formatted_data, gr.update(visible=True), func_gaps_data, gr.update(visible=True), tech_gaps_data, gr.update(visible=True), accept_data, gr.update(visible=False), csv_path

    # Acceptance criteria
    accept_flag, accept = acceptance_criteria(query, testing_cases)
    accept_data = pd.DataFrame(accept)
    yield formatted_data, gr.update(visible=True), func_gaps_data, gr.update(visible=True), tech_gaps_data, gr.update(visible=True), accept_data, gr.update(visible=True), csv_path

    # Generate CSV report at the end
    csv_path = generate_csv_report(query, func_gaps, tech_gaps, testing_cases, accept)
    yield formatted_data, gr.update(visible=True), func_gaps_data, gr.update(visible=True), tech_gaps_data, gr.update(visible=True), accept_data, gr.update(visible=True), csv_path


def export_report(csv_path):
    print(f"CSV report available at: {csv_path}")
    return csv_path

def main():
    with gr.Blocks() as demo:
        gr.Markdown("# EstiMATE")

        requirement_query = gr.Textbox(label="Story Description", lines=5, placeholder="Enter the Story Summary and Description here with all possible details")
        estimate_button = gr.Button("Generate Estimates")

        gapsfunc_table_output = gr.Dataframe(
            interactive=True,
            visible=False,
            label="Functional Gaps Scenarios",
        )

        gapstech_table_output = gr.Dataframe(
            interactive=True,
            visible=False,
            label="Technical Gaps Scenarios",
        )

        test_table_output = gr.Dataframe(
            interactive=True,
            visible=False,
            label="Testing Scenarios",
        )

        accept_table_output = gr.Dataframe(
            interactive=True,
            visible=False,
            label="Acceptance Criterias",
        )


        estimate_button.click(
            process_input,  # Function to call on button click
            inputs=requirement_query,  # Input from the user
            outputs=[test_table_output, test_table_output, gapsfunc_table_output, gapsfunc_table_output,
                     gapstech_table_output, gapstech_table_output,
                     accept_table_output, accept_table_output, gr.Textbox(label="CSV Path", visible=True)]  # Output to display
        )

    demo.launch()

if __name__ == "__main__":
    main()
