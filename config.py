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

# Ensemble NN (Stock Scorer)
N_ENSEMBLE = 20
SCORER_HIDDEN_1 = 32
SCORER_HIDDEN_2 = 16
SCORER_DROPOUT = 0.2
SCORER_EPOCHS = 1500
SCORER_LR = 0.005
SCORER_GRAD_SAMPLE = 0.3

# Monte Carlo Dropout
MC_FORWARD_PASSES = 30

# Stock selection
UNCERTAINTY_PENALTY = 3.0

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
EVENT_RISK_PENALTY = 2.0
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
TRAINING_NN_ARCHITECTURE = [64, 32, 16]  # hidden layers
TRAINING_EPOCHS = 800
TRAINING_LR = 0.0005
TRAINING_HUBER_DELTA = 0.3
