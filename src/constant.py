import os
from enum import Enum

class Constants(Enum):

    PROMPT_PATH = os.path.join("config", "prompt_v2.txt")
    VECSTORE_PATH = "vec_db/vecstores_info.json"
    