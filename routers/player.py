from fastapi import APIRouter

from numpy import *
from typing import List
from models.playerItem import Player
from services.prediction_service import predict_winner


router = APIRouter(prefix="/players", tags=["players"])

@router.post("/predict")
async def getPredictionResults(players : List[Player]): 
    return predict_winner(players)
