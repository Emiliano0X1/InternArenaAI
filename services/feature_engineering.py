import numpy as np

RAW_FEATURES = [
    "easy_count", "med_count", "hard_count",
    "active_days", "recent_active_days", "streak_current",
    "acceptance_ratio", "contest_rating"
]

DERIVED_FEATURES = [
    "hard_ratio", "momentum", "consistency_score"
]

MODEL_FEATURES = [
    "contest_rating", "hard_ratio", "med_ratio", "recent_active_days",
    "streak_current", "acceptance_ratio"
]

def compute_derived(player : dict) -> dict:
    # Soporte para ambas convenciones de nombres (camelCase del modelo Pydantic y snake_case interno)
    total_solved = (player.get("cantEasy", player.get("easy_count", 0))
                    + player.get("cantMed", player.get("med_count", 0))
                    + player.get("cantHard", player.get("hard_count", 0)))
    
    hard_count = player.get("cantHard", player.get("hard_count", 0))
    med_count  = player.get("cantMed",  player.get("med_count",  0))

    hard_ratio = (hard_count / total_solved if total_solved > 0 else 0.0)
    med_ratio  = (med_count  / total_solved if total_solved > 0 else 0.0)

    active_days = player.get("daysActive", player.get("active_days", 0)) or 1
    recent = player.get("recent_active_days", 0)
    momentum = recent / active_days

    consistency_score = player.get("consistency_score", 0.5)

    return {
        "hard_ratio"       : hard_ratio,
        "med_ratio"        : med_ratio,
        "momentum"         : momentum,
        "consistency_score": consistency_score
    }


def enrich_player(player : dict) -> dict:
    derived = compute_derived(player)
    return {
        **player,
        **derived
    }

def load_flat_dataset(matches : list) -> tuple[np.ndarray, np.ndarray]:
    rows, labels = [], []
    for match in matches:
        for p in match["players"]:
            enriched = enrich_player(p)
            rows.append([enriched.get(f) for f in MODEL_FEATURES])
            labels.append(p["won"])
    
    X = np.array(rows, dtype=float)
    Y = np.array(labels, dtype=int)
    return impute_missing(X), Y


def impute_missing(X : np.ndarray) -> np.ndarray:
    X = X.astype(float)
    col_means = np.nanmean(X, axis = 0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])
    return X