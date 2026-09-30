"""Central configuration: paths, constants, and schemas."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORTS_DIR = OUTPUT_DIR / "reports"

for _d in (DATA_DIR, OUTPUT_DIR, MODELS_DIR, PLOTS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_TRAVELERS = 1200
LATENT_DIMS = 12          # SVD dimensions for collaborative filtering
TOP_N_RECOMMENDATIONS = 5

INTERESTS = ["adventure", "culture", "food", "nature", "nightlife",
             "relaxation", "shopping", "history", "wildlife", "photography"]

TRAVELER_TYPES = ["Adventure Seeker", "Cultural Explorer", "Leisure & Relaxation",
                  "Family Vacationer", "Budget Backpacker"]

SEASONS = ["spring", "summer", "autumn", "winter"]
TRANSPORT_OPTIONS = ["flight", "train", "car", "bus"]
ACCOMMODATION_OPTIONS = ["hotel", "resort", "hostel", "homestay", "apartment"]
BUDGET_TIERS = ["low", "medium", "high"]

# Hybrid recommender weights (content / collaborative / segment-affinity)
HYBRID_WEIGHTS = {"content": 0.45, "collaborative": 0.30, "segment": 0.25}