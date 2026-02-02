# ===============================
# Configuration File
# ===============================

# Dataset paths (LOCAL MACHINE)
TRAIN_DIR = "train"
TEST_DIR  = "test"

# Image parameters
IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 64

# Training parameters
EPOCHS = 20
FINE_TUNING_EPOCHS = 10
NUM_CLASSES = 7
EARLY_STOPPING_CRITERIA = 3
SEED = 12

# Class labels
CLASS_LABELS = [
    "Anger", "Disgust", "Fear",
    "Happy", "Neutral", "Sadness", "Surprise"
]
