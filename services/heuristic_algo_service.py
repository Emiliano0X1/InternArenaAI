from typing import List
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from models.playerItem import Player
from services.feature_engineering import enrich_player

WEIGTHS = {
    "hard_ratio"       : 0.20,
    "med_ratio"        : 0.25,
    "acceptanceRatio"  : 0.20,
    "daysActive"       : 0.15,
}

def predict_winner_heuristic(players : List[Player]) -> dict:
    enriched = [enrich_player(p.__dict__) for p in players]
    features = list(WEIGTHS.keys())

    # enrich_player ya calcula hard_ratio y med_ratio; los campos del modelo
    # (acceptanceRatio, daysActive) están disponibles directamente en __dict__
    matrix = np.array([[e.get(f, 0) for f in features] for e in enriched], dtype=float)

    # IMPORTANTE: feature_range=(0.1, 1.0) en lugar del default (0, 1).
    # Con (0, 1), el jugador con el MÍNIMO en TODAS las columnas obtiene
    # exactamente 0 en cada feature → score = 0 (ej: Ana).
    # Con (0.1, 1.0) el mínimo absoluto aporta 0.1, lo que da un score
    # proporcional y justo a la distancia relativa entre jugadores.
    scaler = MinMaxScaler(feature_range=(0.1, 1.0))
    matrix_scaled = scaler.fit_transform(matrix)

    weight_vector = np.array([WEIGTHS[f] for f in features])
    raw_scores = matrix_scaled @ weight_vector

    total = raw_scores.sum()
    probs = raw_scores / total if total > 0 else raw_scores

    result = {p.name: float(prob) for p, prob in zip(players, probs)}
    return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))