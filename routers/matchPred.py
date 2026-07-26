from fastapi import APIRouter  # type: ignore[reportMissingImports]
from typing import Any

from numpy import *
from typing import List
from models.playerItem import Player
from services.heuristic_algo_service import predict_winner_heuristic

router = APIRouter(prefix="/players", tags=["players"])

@router.post("/predict")
async def getPredictionResults(players : List[Player]): 
    return predict_winner_heuristic(players)

#TODO -> Endpoint to get the players data from the LeetcodeAPI and return it to frontend
@router.get("/stats")
async def getStatsFromPlayers(usernames : List[str]):
    pass

#TODO -> Endpoint to store the ranking and is_new when the party is ended
@router.post("/history")
async def storedHistoryData(matchHistory : Any):
    pass
