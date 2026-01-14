#inputs: loads an image and synthesizes a text prompt on the fyl
#loss: it uses symmetric contrastive loss (checking if image matches text) not crossentropy
#tokenizer: it requires the BERT tokenizer to process the text

import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import ast
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from utils import seed_all
from breastclip.data.data_utils import get_density_augmentation
from breastclip.model.mammo_clip import MammoCLIP

class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, image_features, text_features, logit_scale):
        # normalisation
        #divide vectors by their length. Turns dot product into cosine similarity
        image_features = image_features / image_features.norm(dim = 1, keepdim = True)
        text_features = text_features / text_features.norm(dim = 1, keepdim = True)
        
        # similarity matrix (batch x batch)
        #   multiply every image vector by every text vector
        # logit_scale is a learnable 