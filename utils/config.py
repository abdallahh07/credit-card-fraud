from pathlib import Path

# ── Paths ─────────────────────────────────────────────
# config.py is in utils/, so two .parent steps reach the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"

TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
ORIG_FILE = DATA_DIR / "credit_card_fraud_2026.csv"

# ── General ───────────────────────────────────────────
RANDOM_STATE = 42
TARGET = "is_fraud"
ID_COL = "id"
N_SPLITS = 5

# ── Model settings (from the tuning runs) ─────────────
LGBM_PARAMS = {
    "n_estimators": 3000,
    "learning_rate": 0.05,
    "num_leaves": 4,
    "max_depth": 2,
    "min_child_samples": 70,
    "verbose": -1,
    "random_state": RANDOM_STATE,
}

XGB_PARAMS = {
    "n_estimators": 3000,
    "learning_rate": 0.05,
    "max_depth": 2,
    "min_child_weight": 5,
    "subsample": 1.0,
    "colsample_bytree": 0.85,
    "verbosity": 0,
    "random_state": RANDOM_STATE,
}