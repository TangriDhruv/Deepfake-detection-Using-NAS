import os

# General configuration
RANDOM_SEED = 42
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_EVAL = 64
NUM_WORKERS = 4

# Dataset configuration
FEATURE_TYPE = 'mfcc'  # Options: 'mfcc', 'spec', 'cqt'
MAX_SEQ_LEN = 400

# NAS configuration
INPUT_CHANNELS = 60  # 20 MFCCs x 3 (static, delta, delta-delta)
NUM_CELLS = 3
NUM_NODES = 4
NUM_OPS = 10
NAS_EPOCHS = 30
PPO_UPDATES = 5
SEARCH_METHOD = 'hybrid'  # PPO+DARTS

# Training configuration
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 3e-4

# Paths for ASVspoof 2019 LA dataset
BASE_DIR = "/kaggle/input/asvspoof-dataset-2019" #add path to your dataset.
DATA_DIR_TRAIN = os.path.join(BASE_DIR, "LA", "ASVspoof2019_LA_train", "flac")
DATA_DIR_DEV = os.path.join(BASE_DIR, "LA", "ASVspoof2019_LA_dev", "flac")
DATA_DIR_EVAL = os.path.join(BASE_DIR, "LA", "ASVspoof2019_LA_eval", "flac")

TRAIN_PROTOCOL = os.path.join(BASE_DIR, "LA", "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.train.trn.txt")
DEV_PROTOCOL = os.path.join(BASE_DIR, "LA", "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.dev.trl.txt")
EVAL_PROTOCOL = os.path.join(BASE_DIR, "LA", "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.eval.trl.txt")

# WandB configuration
WANDB_PROJECT = "ASVspoof2019-NAS"
WANDB_API_KEY = ""  # Add your API key here or use environment variable

# Output directory
OUTPUT_DIR = "./results"
