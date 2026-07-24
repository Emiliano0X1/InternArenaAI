# InternArenaAI — Predictions & Mathematical Analysis Service

> **FastAPI microservice** that evaluates player performance stats and predicts who has the highest probability of winning a LeetCode Arena match using custom mathematical algorithms.

---

## 📐 High-Level Architecture

```
HTTP Client (Frontend / Other Microservice)
        │
        │  POST /players/predict
        ▼
┌─────────────────────────────────────────────────────────┐
│                        main.py                          │
│              FastAPI App (entry point)                  │
│         app.include_router(router, prefix="/players")   │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│              routers/player.py                          │
│    APIRouter  prefix="/players"  tags=["players"]       │
│                                                         │
│    POST /predict  →  predict_winner_heuristic()         │
└───────────────────┬─────────────────────────────────────┘
                    │  List[Player]  (validated by Pydantic)
                    ▼
┌─────────────────────────────────────────────────────────┐
│         services/heuristic_algo_service.py              │
│                                                         │
│  1. enrich_player()   — adds computed ratios            │
│  2. MinMaxScaler      — normalizes feature matrix       │
│  3. weighted dot      — calculates raw scores           │
│  4. softmax-style     — converts to probabilities       │
└───────────────────┬─────────────────────────────────────┘
                    │  calls
                    ▼
┌─────────────────────────────────────────────────────────┐
│         services/feature_engineering.py                 │
│                                                         │
│  compute_derived()  — hard_ratio, med_ratio, momentum   │
│  enrich_player()    — merges raw + derived features     │
└─────────────────────────────────────────────────────────┘
```

---

## 🗂️ Project Structure

```
InternArenaAI/
│
├── main.py                          # FastAPI app entry point
│
├── models/
│   └── playerItem.py                # Pydantic data model (input validation)
│
├── routers/
│   └── player.py                    # HTTP routes / controllers
│
├── services/
│   ├── heuristic_algo_service.py    # ★ Main prediction engine (active)
│   ├── feature_engineering.py      # Feature computation & enrichment
│   ├── prediction_service.py       # Linear regression engine (alternative)
│   └── gauss_logic.py              # Gaussian elimination math solver
│
└── utils/
    └── math_func.py                 # Low-level math helpers (sum, multiply, power)
```

---

## 🔄 Complete Request Flow (Step by Step)

This is exactly what happens when `POST /players/predict` is called with a JSON array of players.

```
Step 1 ── HTTP Layer
│
│   Client sends:
│   POST http://localhost:8000/players/players/predict
│   Body: [ { "name": "carlos", "cantEasy": 30, ... }, ... ]
│
▼
Step 2 ── Pydantic Validation  [models/playerItem.py]
│
│   FastAPI automatically deserializes the JSON array into List[Player].
│   Each Player object is validated:
│     - name          → str
│     - cantEasy      → int   (number of easy problems solved)
│     - cantMed       → int   (number of medium problems solved)
│     - cantHard      → int   (number of hard problems solved)
│     - score         → int   (current accumulated score)
│     - daysActive    → int   (how many days the player has been active)
│     - acceptanceRatio → float  (accepted / total submissions)
│     - medRatio      → float  (medium solved / total solved)
│
│   If any field is missing or has the wrong type → 422 Unprocessable Entity.
│
▼
Step 3 ── Router dispatches  [routers/player.py]
│
│   predict_winner_heuristic(players) is called with the validated list.
│
▼
Step 4 ── Feature Enrichment  [services/feature_engineering.py]
│
│   For each player, enrich_player(p.__dict__) is called.
│   This runs compute_derived(), which calculates:
│
│     total_solved  = cantEasy + cantMed + cantHard
│     hard_ratio    = cantHard / total_solved
│     med_ratio     = cantMed  / total_solved
│     momentum      = recent_active_days / daysActive  (0 if not provided)
│     consistency_score = 0.5 (default, used for ML model)
│
│   The result is a dict merging the original fields + new derived ones.
│
▼
Step 5 ── Feature Matrix Construction  [services/heuristic_algo_service.py]
│
│   Only 4 features are selected from each enriched player:
│     WEIGHTS = {
│       "hard_ratio"      : 0.20,
│       "med_ratio"       : 0.25,
│       "acceptanceRatio" : 0.20,
│       "daysActive"      : 0.15,
│     }
│
│   A NumPy matrix is built:  shape = (n_players, 4)
│     matrix[0] = [carlos.hard_ratio, carlos.med_ratio, ...]
│     matrix[1] = [juan.hard_ratio,   juan.med_ratio,   ...]
│     matrix[2] = [ana.hard_ratio,    ana.med_ratio,    ...]
│
▼
Step 6 ── Normalization with MinMaxScaler
│
│   MinMaxScaler(feature_range=(0.1, 1.0)) is applied COLUMN-WISE.
│
│   Formula per feature column:
│     x_scaled = 0.1 + 0.9 * (x - col_min) / (col_max - col_min)
│
│   WHY 0.1 minimum instead of 0?
│   If a player is the worst in every column, their scaled value would be
│   0 in each column → dot product with weights = 0 (unfair absolute zero).
│   Using 0.1 as floor ensures the weakest player still gets a
│   proportional non-zero score.
│
▼
Step 7 ── Weighted Score (Dot Product)
│
│   raw_score[i] = matrix_scaled[i] · weight_vector
│
│   Expanded:
│     score = (hard_ratio_scaled  × 0.20)
│           + (med_ratio_scaled   × 0.25)
│           + (acceptance_scaled  × 0.20)
│           + (daysActive_scaled  × 0.15)
│
▼
Step 8 ── Probability Normalization
│
│   total = sum of all raw_scores
│   prob[i] = raw_score[i] / total
│
│   This converts absolute weighted scores into relative probabilities
│   that sum to 1.0 (100%).
│
▼
Step 9 ── Sort & Return
│
│   Players are sorted descending by probability.
│   Response:
│     { "carlos": 0.572, "juan": 0.370, "ana": 0.057 }
```

---

## 🧮 Core Concepts & Terminology

### Pydantic Model (Validation Layer)
> **File:** `models/playerItem.py`

Pydantic is a Python library for **data validation using type hints**. When FastAPI receives a JSON payload, it passes it through the Pydantic model automatically.

```python
class Player(BaseModel):
    name          : str
    cantEasy      : int     # Easy problems solved
    cantMed       : int     # Medium problems solved
    cantHard      : int     # Hard problems solved
    score         : int     # Current score (used in regression service)
    daysActive    : int     # Days the player has been active
    acceptanceRatio : float # Ratio: accepted submissions / total attempts
    medRatio      : float   # Ratio: medium solved / total solved (sent by client)
```

> [!NOTE]
> `medRatio` is sent directly by the client in the request. The `compute_derived()` function **also** computes it internally from `cantMed / total_solved` as a cross-check. The internally computed value is used by the heuristic engine.

---

### Feature Engineering
> **File:** `services/feature_engineering.py`

**Feature Engineering** is the process of transforming raw data into features that better represent the underlying patterns for the prediction model.

#### `compute_derived(player: dict) → dict`
Computes derived (calculated) features from the raw input fields:

| Feature | Formula | What it measures |
|---|---|---|
| `hard_ratio` | `cantHard / total_solved` | Proportion of hard problems attempted — indicates skill level |
| `med_ratio` | `cantMed / total_solved` | Proportion of medium problems — balance indicator |
| `momentum` | `recent_active_days / daysActive` | How recently active (0 if not provided) |
| `consistency_score` | `0.5` (default) | Placeholder for future consistency metric |

#### `enrich_player(player: dict) → dict`
Merges the original raw fields with the computed derived fields into a single dictionary. This is the "enriched" representation used downstream.

---

### MinMaxScaler (Normalization)
> **Library:** `scikit-learn`  **File:** `services/heuristic_algo_service.py`

**Normalization** brings all features to the same numeric scale so that features with large values (like `daysActive = 90`) don't dominate features with small values (like `hard_ratio = 0.25`).

**MinMaxScaler formula** (applied per column):

$$x_{scaled} = min_{range} + (max_{range} - min_{range}) \cdot \frac{x - x_{col\_min}}{x_{col\_max} - x_{col\_min}}$$

With `feature_range=(0.1, 1.0)`:
- The **worst** player in a column → `0.1`
- The **best** player in a column → `1.0`
- Everyone else → proportionally between `0.1` and `1.0`

> [!WARNING]
> With the default `feature_range=(0, 1)`, a player who is the minimum in **all** columns would receive a score of exactly `0` even though they may have decent stats in absolute terms. The `(0.1, 1.0)` range prevents this edge case.

---

### Weighted Dot Product (Scoring)
> **File:** `services/heuristic_algo_service.py`

After normalization, each player's score is computed as a **weighted sum** of their scaled features:

$$\text{score}_i = \sum_{f} w_f \cdot x_{i,f}^{scaled}$$

Expanded with current weights:

$$\text{score}_i = (0.20 \times \text{hard\_ratio}) + (0.25 \times \text{med\_ratio}) + (0.20 \times \text{acceptanceRatio}) + (0.15 \times \text{daysActive})$$

**Why these weights?** They encode business knowledge:
- `med_ratio` gets the highest weight (0.25) because medium problems are a good balance of skill and volume.
- `hard_ratio` and `acceptanceRatio` share 0.20 each — both signal quality over quantity.
- `daysActive` (0.15) rewards consistency but less than skill metrics.

> [!TIP]
> The weights are defined in the `WEIGTHS` dict at the top of `heuristic_algo_service.py`. You can tune them to change how the model ranks players.

---

### Probability Normalization (Softmax-style)
> **File:** `services/heuristic_algo_service.py`

After computing raw weighted scores, they are converted to **relative probabilities**:

$$P_i = \frac{\text{score}_i}{\sum_{j} \text{score}_j}$$

This ensures the output probabilities always sum to `1.0`, making them interpretable as *"probability of winning"* rather than raw scores.

---

### Gaussian Elimination (Alternative Engine)
> **File:** `services/gauss_logic.py`

This module implements **Gaussian Elimination**, a classic numerical method from linear algebra for solving systems of linear equations of the form:

$$A \cdot \mathbf{x} = \mathbf{b}$$

It is used by `prediction_service.py` (the alternative regression-based engine, **not the active route**) to solve the **Normal Equations** of Multiple Linear Regression:

$$\begin{pmatrix} n & \Sigma x_1 & \Sigma x_2 & \Sigma x_3 \\ \Sigma x_1 & \Sigma x_1^2 & \Sigma x_1 x_2 & \Sigma x_1 x_3 \\ \vdots & & \ddots & \vdots \end{pmatrix} \begin{pmatrix} b_0 \\ b_1 \\ b_2 \\ b_3 \end{pmatrix} = \begin{pmatrix} \Sigma y \\ \Sigma x_1 y \\ \vdots \end{pmatrix}$$

#### How `gaussGetVariables(arr)` works:

```
Phase 1: Forward Elimination
  For each pivot row, eliminate all entries below the diagonal
  by subtracting a scaled version of the pivot row.
  Result: Upper triangular matrix.

Phase 2: Back Substitution
  Starting from the last row (one unknown), solve each equation
  substituting already-known values upward.
  Result: The coefficient vector [b0, b1, b2, b3]
```

---

### Math Utility Functions
> **File:** `utils/math_func.py`

Low-level helpers used by `prediction_service.py` to build the Normal Equations matrix:

| Function | Signature | Purpose |
|---|---|---|
| `sumElements` | `(row) → float` | $\sum x_i$ — sum of all elements in a column |
| `multiplyElements` | `(row1, row2) → float` | $\sum x_i y_i$ — element-wise multiply then sum |
| `powerElement` | `(row) → float` | $\sum x_i^2$ — sum of squared elements |
| `normalize` | `(arr) → np.ndarray` | Z-score normalization: $(x - \mu) / \sigma$ |
| `ratioMedium` | `(cantMed, total) → float` | Simple division ratio |
| `calculateScore` | `(easy, med, hard) → int` | `easy + (med × 3) + (hard × 5)` |

---

## 🔀 Two Prediction Engines (Active vs. Alternate)

The project has **two independent prediction algorithms**. Currently only the heuristic one is wired to the router.

| | Heuristic Engine | Regression Engine |
|---|---|---|
| **File** | `heuristic_algo_service.py` | `prediction_service.py` |
| **Active?** | ✅ Yes (used by `/predict`) | ❌ No (not routed) |
| **Algorithm** | Weighted scoring + normalization | Multiple Linear Regression via Gauss |
| **Requires `score`?** | No | Yes — uses `score` as the target `Y` variable |
| **Input** | Raw player stats | Player stats + historical score |
| **Output** | Win probabilities (sum = 1) | Predicted score per player |
| **Math** | Dot product + MinMaxScaler | OLS Normal Equations + Gauss Elimination |

> [!NOTE]
> The regression engine (`prediction_service.py`) was the original design. It requires `score ≠ 0` for all players to fit the regression, which is why it was replaced by the heuristic engine for the current use case where scores start at `0`.

---

## 🚀 Getting Started

### Prerequisites
- **Python** 3.9+
- **pip**

### 1. Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install fastapi uvicorn numpy scikit-learn sympy pydantic
```

### 3. Run the Server

```bash
uvicorn main:app --reload
```

Server runs at **`http://localhost:8000`**

| URL | Description |
|---|---|
| `http://localhost:8000/docs` | Swagger UI — interactive API explorer |
| `http://localhost:8000/redoc` | ReDoc — clean API docs |

> [!NOTE]
> The full prediction endpoint URL is `POST http://localhost:8000/players/players/predict`.
> The double `/players/players/` is because the prefix is declared **twice**: once in `main.py` (`prefix="/players"`) and once in `routers/player.py` (`prefix="/players"`).

---

## 📡 API Reference

### `POST /players/players/predict`

Predict the win probability ranking for a list of players.

**Request Body:** `application/json` — Array of Player objects.

```json
[
  {
    "name": "carlos",
    "cantEasy": 30,
    "cantMed": 35,
    "cantHard": 22,
    "score": 0,
    "daysActive": 90,
    "acceptanceRatio": 0.81,
    "medRatio": 0.40
  },
  {
    "name": "juan",
    "cantEasy": 45,
    "cantMed": 30,
    "cantHard": 15,
    "score": 0,
    "daysActive": 60,
    "acceptanceRatio": 0.78,
    "medRatio": 0.40
  }
]
```

**Response:** `200 OK`

```json
{
  "carlos": 0.5724621524992275,
  "juan": 0.3702916322508497,
  "ana": 0.0572462152499227
}
```

Values are **win probabilities** between `0` and `1`, sorted descending. They always sum to `1.0`.

---

## 🗝️ Glossary

| Term | Definition |
|---|---|
| **FastAPI** | Modern Python web framework for building APIs with automatic data validation and OpenAPI docs generation |
| **Pydantic** | Python library for data validation using type annotations; used by FastAPI to validate request bodies |
| **Uvicorn** | ASGI server (like Apache/Nginx but for async Python); used to run the FastAPI app |
| **APIRouter** | FastAPI class that groups related routes; like a sub-application that gets mounted on the main app |
| **MinMaxScaler** | scikit-learn transformer that scales features to a fixed range (e.g., `[0.1, 1.0]`) column by column |
| **Feature** | An individual measurable property of the data used as input to the model (e.g., `hard_ratio`) |
| **Feature Engineering** | The process of creating new features from raw data to improve model performance |
| **Normalization** | Rescaling features so they are on the same numeric scale, preventing larger-magnitude features from dominating |
| **Weighted Score** | A score computed as the sum of features multiplied by their respective importance weights |
| **Dot Product** | Mathematical operation: $\mathbf{a} \cdot \mathbf{b} = \sum a_i b_i$ — multiplies elements pairwise and sums the result |
| **Multiple Linear Regression** | A statistical model predicting `Y` from multiple `X` variables: $Y = b_0 + b_1 X_1 + b_2 X_2 + \ldots$ |
| **OLS (Ordinary Least Squares)** | Method that finds regression coefficients by minimizing the sum of squared differences between predicted and actual values |
| **Normal Equations** | A system of linear equations derived from OLS that directly gives the optimal regression coefficients |
| **Gaussian Elimination** | Numerical method for solving systems of linear equations; transforms a matrix to row echelon form via forward elimination then back substitution |
| **NumPy** | Python library for numerical computing; provides N-dimensional arrays and math operations |
| **scikit-learn** | Python machine learning library; provides tools like `MinMaxScaler` and preprocessing utilities |
| **`__dict__`** | Python built-in attribute that returns all instance attributes as a dictionary; used to convert a Pydantic `Player` object into a plain dict |
| **Microservice** | An independent, single-responsibility service that communicates via HTTP; this repo is the AI/ML microservice for LeetArena |
