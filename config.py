"""
config.py — hyperparameters and settings
"""

# Portfolio
TOTAL_CAPITAL_KRW = 20_000_000
N_SELECT = 5

TIME_MARGINALS = [0.15, 0.40, 0.45]
RISK_MARGINALS = [0.45, 0.40, 0.15]

TIME_LABELS = ["Short(~3mo)", "Mid(~1yr)", "Long(1yr+)"]
RISK_LABELS = ["Aggressive", "Balanced", "Stable"]

TIME_MULTIPLIERS = [0.40, 0.75, 0.95]
RISK_TIER_MIDPOINTS = [0.35, 0.15, 0.05]

# Ensemble NN
N_ENSEMBLE = 20
EARLY_STOP_PATIENCE = 41   # patience for val-loss based early stopping

# Monte Carlo Dropout
MC_FORWARD_PASSES = 30

# Stock selection
UNCERTAINTY_PENALTY = 1.0

# Matrix Network
MATRIX_HIDDEN_1 = 64
MATRIX_HIDDEN_2 = 32
MATRIX_OUTPUT = 9

# End-to-End Training
E2E_EPOCHS = 500
E2E_LR_MAX = 0.003
E2E_LR_MIN = 0.0001
E2E_NOISE_STD = 0.008

SINKHORN_ITERS = 80

LOSS_AVERSION = 2.5
LAMBDA_SHARPE = 5.0
LAMBDA_RISK = 3.0
LAMBDA_CONCENTRATION = 6.0
LAMBDA_ENTROPY = 0.5
LAMBDA_MARGINAL = 200.0
MAX_CELL_ALLOCATION = 0.25

# Adam Optimizer
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8

# Random Seed
RANDOM_SEED = 42

# Screener
SCREENER_MIN_VOLUME_USD = 1_000_000
SCREENER_MIN_HISTORY_DAYS = 126
SCREENER_MAX_UNIVERSE = 50

# Sentiment
SENTIMENT_MODEL = 'finbert'            # 'finbert' or 'vader'
SENTIMENT_LOOKBACK_DAYS = 30
SENTIMENT_WEIGHT_IN_SCORE = 0.10
EVENT_RISK_PENALTY = 0.0
FINNHUB_API_KEY = ''                   # get a free key at https://finnhub.io

# FRED API
FRED_API_KEY = ''                      # get a free key at https://fred.stlouisfed.org

# Training Universe
TRAINING_USE_SP500 = True              # use full S&P 500 for training
TRAINING_USE_NASDAQ100 = True          # also include NASDAQ-100

# --- Ablation settings (Task #4) ---
# Set these to False to exclude corresponding feature groups during training.
# Used for ablation studies to measure each component's contribution.
USE_MACRO_FEATURES = True      # FRED (13) + Fama-French (15) + cross-asset (15)
USE_SENTIMENT_FEATURES = True  # FinBERT + SEC + FDA + earnings (22 features)

TRAINING_PERIOD = '10y'                # training data period
TRAINING_SNAPSHOT_INTERVAL = 10        # snapshot interval (trading days)
# v2.3.15 NLL search Trial 52 (v2.3.16 verdict: IMPROVEMENT, Amendment 1)
# 60-trial TPE search; Stage 1 rank_corr 0.5732 (N=5); Stage 2 N=20 0.5371
TRAINING_NN_ARCHITECTURE = [128, 64, 32]  # 'large' (Trial 52)
TRAINING_EPOCHS = 20000   # [v2.3.15 Amendment 2] 5000 → 20000 (Prechelt 2.0× safety margin under NLL convergence)
TRAINING_LR = 0.001563563963064687           # ~1.56e-3
TRAINING_WEIGHT_DECAY = 4.4856093488331435e-05  # ~4.49e-5
TRAINING_HUBER_DELTA = 0.5
TRAINING_DROPOUT = 0.1146035891599349                  # v2.3.15 Trial 52 (tunable in v2.3.15+)

# Feature selection thresholds
VAR_THRESHOLD = 0.0010819081885486052            # ~0.00108
CORR_THRESHOLD = 0.05584259829572068             # ~0.0558

# SEC EDGAR User-Agent (required by SEC fair-use policy)
# Format: 'Real Name email@domain' — fill in locally, keep empty on GitHub
SEC_USER_AGENT = ''
