import numpy as np
from datetime import datetime, timezone

RAW_FEATURES = [
    "easy_count", "med_count", "hard_count",
    "active_days", "recent_active_days", "streak_current",
    "acceptance_ratio", "contest_rating"
]

DERIVED_FEATURES = [
    "hard_ratio", "momentum", "consistency_score"
]

MODEL_FEATURES = [
    "ranking", "hard_ratio", "med_ratio", "momentum",
    "acceptance_ratio", "is_new"
]

def compute_derived(player : dict) -> dict:
    #1. Get total for all the next operations
    total_solved = (player.get("cantEasy")
                    + player.get("cantMed")
                    + player.get("cantHard"))


    #2. Get Hard ratio
    hard_count = player.get("cantHard")
    hard_ratio = (hard_count / total_solved if total_solved > 0 else 0.0)

    #3. Get Med ratio
    med_count  = player.get("cantMed")
    med_ratio  = (med_count  / total_solved if total_solved > 0 else 0.0)

    # 4. Get Momentum (days active recently)
    now = datetime.now(timezone.utc).timestamp()
    last_active = player.get("last_active")

    MAX_DAYS = 10

    if last_active:
        days_since = max(0.0, (now - last_active) / 86400)
        momentum = max(0.0, 1.0 - (days_since / MAX_DAYS))
    else:
        momentum = 0.0


    # 5. Calculate acceptance_ratio
    recent_submissions = player.get("recent_submissions", [])

    N = len(recent_submissions)

    if N > 0:
        accepted_recent = sum(1 for sub in recent_submissions if sub == 'ACCEPTED')

        k = 3.0
        global_avg = 0.50
        acceptance_ratio = (accepted_recent + (k * global_avg)) / (N + k)
    else:
        acceptance_ratio = 0.25

    return {
        "hard_ratio"       : hard_ratio,
        "med_ratio"        : med_ratio,
        "momentum"         : momentum,
        "acceptance_ratio": acceptance_ratio
    }


def enrich_player(player : dict) -> dict:
    derived = compute_derived(player)
    return {
        **player,
        **derived
    }



""" def load_flat_dataset(matches : list) -> tuple[np.ndarray, np.ndarray]:
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
    return X """
