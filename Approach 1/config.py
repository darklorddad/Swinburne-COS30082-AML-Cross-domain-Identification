import torch
import os

# Dataloader and Dataset Configurations 
DATA_DIR = '.' 
TRAIN_LIST = 'list/train.txt'
TEST_LIST = 'list/groundtruth.txt' # We use this file because it has the labels for the test images.
DATA_MODE = 'all' 
CLASS_MODE = 'mixed' # Options: 'with_pairs', 'without_pairs', 'mixed' (all classes)

# Model Configurations
# CHOOSE YOUR MODEL HERE: 'resnet50', 'convnextv2', or 'xception'
MODEL_NAME = 'resnet50' 

NUM_CLASSES = 100 

# Model save paths
MODEL_SAVE_PATH_RESNET50 = 'resnet50_model.pth'
MODEL_SAVE_PATH_CONVNEXTV2 = 'convnextv2_model.pth'
MODEL_SAVE_PATH_XCEPTION = 'xception_model.pth'

# --- Training Hyperparameters ---
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
IMAGE_SIZE = 256 # Higher resolution for better detail (Standard is 224)
# Set to 0 if you have issues on Windows, otherwise 4 is good
NUM_WORKERS = 0 if os.name == 'nt' else 4
EARLY_STOPPING_PATIENCE = 10 # Number of epochs to wait for improvement before stopping

# --- Learning Rate Schedulers ---
# For ReduceLROnPlateau (ResNet/Xception)
SCHEDULER_PATIENCE = 5 # Epochs to wait for improvement before reducing LR
SCHEDULER_FACTOR = 0.1 # Factor by which to reduce LR

# For CosineAnnealingWarmRestarts (ConvNeXt)
T_0 = 10      # Number of epochs for the first restart
T_MULT = 2    # A factor increases T_i after a restart

MIN_LR = 1e-6 # Minimum learning rate

# --- Advanced Training Settings ---
WEIGHT_DECAY = 0.0001 # Regularization parameter for AdamW (Reduced from 0.01)
LABEL_SMOOTHING = 0.1 # Reduces overfitting by preventing 100% confidence
DROP_PATH_RATE = 0.2 # Stochastic Depth rate (Regularization for deep models)