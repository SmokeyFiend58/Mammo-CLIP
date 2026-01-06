#new file for training on density/ BI-RADS grading.
import warnings
import argparse
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter

from utils import seed_all
from breastclip.model.modules.image_encoder import SwinTransformer_Mammo

from breastclip.data.data_utils import load_transform

warnings.filterwarnings("ignore")

class MultiHeadSwin(nn.Module):
    def __init__(self, encoder_name, img_size, num_density = 4, num_birads = 5):
        super().__init__()

    
    #load swin backbone
    
        self.encoder = SwinTransformer_Mammo(
            name = encoder_name,
            pretrained= True
            img_size= img_size
        )
        inputDim = self.encoder.outDim

        #head 1 density
        self.head_density = nn.Linear(inputDim, num_density)
        
        # head 2 birads
        self.head_birads = nn.Linear(inputDim, num_birads)
        
    def forward(self, x):
        features = self.encoder(x)
        dOut = self.head_density(features)
        bOut = self.head_birads(features)
        
        return dOut, bOut
    
#dataset handler

