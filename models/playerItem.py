from pydantic import BaseModel

class Player(BaseModel):
    name : str
    cantEasy : int
    cantMed : int
    cantHard : int
    score : int
    daysActive : int
    acceptanceRatio : float
    medRatio : float



