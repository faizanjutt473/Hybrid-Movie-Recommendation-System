"""Central configuration for the recommendation system project."""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ml-100k"
RESULTS_DIR = PROJECT_ROOT / "results"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42

# --------------------------------------------------------------------------- #
# Data split (per-user, time based)
# --------------------------------------------------------------------------- #
TEST_RATIO = 0.2          # newest 20% of every user's ratings -> test set
VAL_RATIO = 0.2           # newest 20% of the *train* ratings -> validation set
MIN_USER_RATINGS = 5      # users with fewer ratings stay entirely in train

# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
RELEVANCE_THRESHOLD = 4.0     # a test rating >= 4 counts as a relevant item
TOP_K_VALUES = (5, 10, 20)    # K values reported in the results table
PRIMARY_K = 10                # K used for hybrid tuning and for plots

# --------------------------------------------------------------------------- #
# Model hyper-parameters
# --------------------------------------------------------------------------- #
SVD_FACTORS = 50              # latent factors of the (Pure)SVD model
KNN_NEIGHBORS = 50            # neighbours kept per item in item-based CF
KNN_SHRINKAGE = 10.0          # similarity shrinkage (damps low co-rating counts)
LIKE_THRESHOLD = 4.0          # ratings >= 4 build the content-based user profile
BAYESIAN_QUANTILE = 0.5       # "minimum votes" quantile for weighted popularity

# --------------------------------------------------------------------------- #
# Hybrid model / cold start
# --------------------------------------------------------------------------- #
WEIGHT_GRID_STEP = 0.1        # granularity of the hybrid weight search
COLD_START_MIN_RATINGS = 20   # ratings needed before a user is treated as "warm"
COLD_START_PRIOR_STRENGTH = 10.0  # pseudo-ratings of trust in the popularity fallback
COLD_START_GENRE_WEIGHT = 0.7  # genre-onboarding share vs. popularity (new user)
COLD_START_REVEAL = (0, 1, 3, 5, 10, 20)  # ratings revealed in the cold-start test
