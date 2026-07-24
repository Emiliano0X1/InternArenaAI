from fastapi import APIRouter  # type: ignore[reportMissingImports]

from numpy import *
from typing import List
from models.playerItem import Player
from services.heuristic_algo_service import predict_winner_heuristic


router = APIRouter(prefix="/players", tags=["players"])

@router.post("/predict")
async def getPredictionResults(players : List[Player]): 
    return predict_winner_heuristic(players)
