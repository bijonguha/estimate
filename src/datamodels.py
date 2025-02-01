from pydantic import BaseModel, Field
from typing import List, Optional

# LLM Query data models
class Subtasks(BaseModel):
    name: str = Field(description="Name of the subtasks for implementation")
    description: str = Field(description="Detailed description of the subtasks for implementation")

class Testingdetails(BaseModel):
    name: str = Field(description="Name of the unit testing task for testing of subtask implementation")
    description: str = Field(description="Detailed description of the unit testing task for testing of subtask implementation")

class ImportantInstructions(BaseModel):
    name: str = Field(description="name of Important instructions and corner cases for the robustness of the developed feature")
    description : str = Field(description="Detailed description of the important instructions and corner cases for the robustness of the developed feature")

class FeatureResponse(BaseModel):
    subtasks_details: List[Subtasks] = Field(description="List of Sub tasks for implementation")
    unit_testing_details: List[Testingdetails] = Field(description="List of Unit Testing details for the feature where unit test details need to be provided for the subtasks")
    important_instructions: List[ImportantInstructions] = Field(description="List of Important instructions and corner cases for the robustness of the developed feature")

# Request Response data models
class QuerySimple(BaseModel):
    query : str
    vectorstore_id : Optional[str] = None

class QueryFeature(BaseModel):
    query : str
    vectorstore_id : str

class ResponseFile(BaseModel):
    vectorstore_id : str