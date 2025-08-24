from pydantic import BaseModel

class Player(BaseModel):
    name : str
    cantMed : int
    cantHard : int
    time : float
    percentageOfWin : float