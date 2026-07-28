from typing import List
from pydantic import BaseModel

class Player(BaseModel):
    name : str
    cantEasy : int
    cantMed : int
    cantHard : int
    last_active : int
    recent_submissions : List[str]

