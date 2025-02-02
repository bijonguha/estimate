import mlflow
import json
import time
import tempfile

class MLflowLogger:
    def __init__(self, tracking_uri, experiment_name="FastAPI_Logging"):
        """
        Initialize the MLflow logger.
        :param experiment_name: Name of the MLflow experiment
        """
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        
    def log_query(self, query: str, prompt: str, retriever: list, llm_resp: dict = None):
        """
        Logs a query, prompt, and a list of retrieved documents to MLflow within a temporary run.
        :param query: User query
        :param prompt: Model-generated prompt
        :param retriever: List of retrieved documents
        :param metadata: Additional metadata (optional)
        """
        with mlflow.start_run():
            start_time = time.time()  # Track execution time
            
            # Log basic parameters
            mlflow.log_param("query", query)
            mlflow.log_param("prompt", prompt)

            # Log retriever as JSON (since it's a list)
            retriever_json = json.dumps(retriever, indent=4)
            mlflow.log_param("retriever_count", len(retriever))  # Log the number of retrieved docs
            mlflow.set_tag("retriever_data", retriever_json)  # Store JSON as a tag

            # Add MLflow tags
            mlflow.set_tag("query_tag", f"Query: {query}")
            mlflow.set_tag("prompt_tag", f"Prompt: {prompt}")
            
            # Log metadata
            if llm_resp:
                mlflow.log_param("llm_response", json.dumps(llm_resp))

            # Log execution time as a metric
            execution_time = time.time() - start_time
            mlflow.log_metric("execution_time", execution_time)

            # Save log as an artifact (JSON)
            log_data = {
                "query": query,
                "prompt": prompt,
                "retriever": retriever,  # List of retrieved documents
                "llm_response": llm_resp,
                "mlflow_execution_time": execution_time
            }
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as temp_file:
                json.dump(log_data, temp_file, indent=4)
                temp_file_path = temp_file.name
            
            mlflow.log_artifact(temp_file_path, artifact_path="logs")

        print(f"Logged query: {query} and ended run.")
