'''
TRAIN_DIR = "datasets\horse2zebra\horse2zebra"
VAL_DIR = "datasets\horse2zebra\horse2zebra"
'''

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "DATA_PNG/train"
VAL_DIR = "DATA_PNG/val"
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 150
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN_DWI = "genDWI.pth.tar"
CHECKPOINT_GEN_FLAIR = "genFLAIR.pth.tar"
CHECKPOINT_CRITIC_DWI = "criticDWI.pth.tar"
CHECKPOINT_CRITIC_FLAIR = "criticFLAIR.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        # A.ColorJitter(p=0.1),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)
