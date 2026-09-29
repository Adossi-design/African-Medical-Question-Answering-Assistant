"""Paths and fixed settings shared by the notebooks and the web app."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_CSV = ROOT / "data" / "raw" / "afri_med_qa_15k_v2.5_phase_2_15275.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
PROBE_DIR = ROOT / "data" / "probes"
EXTERNAL_DIR = ROOT / "data" / "external"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

SEED = 42             # used for every random operation so results can be reproduced
VAL_FRACTION = 0.15   # share of the official *train* portion held out for validation

BASE_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
