# EstiMATE

Gen-AI tool to estimate efforts for any use case. Additionally, it can be plugged into different agile boards available for seamless management.

## Description

EstiMATE is a Gen-AI powered tool designed to estimate the effort required for various use cases. It leverages large language models to analyze use case descriptions and provide effort estimations. The tool can be integrated with different agile boards for seamless project management.

## Installation

1.  Clone the repository:

    ```bash
    git clone <repository_url>
    ```
2.  Navigate to the project directory:

    ```bash
    cd estimate
    ```
3.  Create a virtual environment:

    ```bash
    python -m venv esti_env
    ```
4.  Activate the virtual environment:

    *   On Windows:

        ```bash
        .\esti_env\Scripts\activate
        ```
    *   On macOS and Linux:

        ```bash
        source esti_env/bin/activate
        ```
5.  Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Set the environment variables:

    Create a `.env` file in the project root directory and set the necessary environment variables. Example:

    ```
    AZUREOPENAI_API_KEY=<your_openai_api_key>
    ```

2.  Run the application:

    ```bash
    uvicorn main:app
    ```

## File Descriptions

*   `.dockerignore`: Specifies intentionally untracked files that Git should ignore.
*   `.gitignore`: Specifies intentionally untracked files that Git should ignore.
*   `Dockerfile`: A text file that contains all the commands a user could call on the command line to assemble an image.
*   `grad_app_fapi.py`: Gradio application file.
*   `LICENSE`: Contains the license information for the project.
*   `main.py`: The main entry point for the application.
*   `README.md`: This file, providing information about the project.
*   `requirements.txt`: Contains a list of Python packages required to run the application.
*   `start`: A script to start the application.

## Directory Descriptions

*   `notebooks/`: Contains Jupyter notebooks.
*   `src/`: Contains the source code for the application.
*   `src/prompts_template/`: Contains prompt templates.
*   `vec_db/`: Contains the vector database.
