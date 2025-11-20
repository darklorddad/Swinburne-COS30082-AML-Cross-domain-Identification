import torch
import os

# Dataloader and Dataset Configurations 
DATA_DIR = '.' 
TRAIN_LIST = 'list/train.txt'
TEST_LIST = 'list/groundtruth.txt' # We use this file because it has the labels for the test images.

# Model Configurations
# CHOOSE YOUR MODEL HERE: 'resnet50', 'convnextv2', or 'xception'
MODEL_NAME = 'xception' 

NUM_CLASSES = 100 

# Model save paths
MODEL_SAVE_PATH_RESNET50 = 'resnet50_mixstream_baseline.pth'
MODEL_SAVE_PATH_CONVNEXTV2 = 'convnextv2_model.pth'
MODEL_SAVE_PATH_XCEPTION = 'xception_model.pth'

# --- Training Hyperparameters ---
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
# Set to 0 if you have issues on Windows, otherwise 4 is good
NUM_WORKERS = 0 if os.name == 'nt' else 4
EARLY_STOPPING_PATIENCE = 5 # Number of epochs to wait for improvement before stopping

# --- Learning Rate Scheduler ---
SCHEDULER_PATIENCE = 2 # Epochs to wait for improvement before reducing LR
SCHEDULER_FACTOR = 0.1 # Factor by which to reduce LR
MIN_LR = 1e-6 # Minimum learning rate