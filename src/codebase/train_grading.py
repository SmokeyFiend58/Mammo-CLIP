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
            pretrained= True,
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
class VinDrSwinDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform = None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        
        #map density from letters to numbes
        self.density_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        
        #map Birads
        self.birads_map = {1: 0, 2:1, 3:2, 4:3, 5:4}
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        #load image
        imgID = f"{row['image_id']}.png"
        studyFolder = row['study_id']
        
        img_path = os.path.join(self.img_dir, studyFolder, imgID)
        
        image = Image.open(img_path).convert("RGB")
        
        #apply augmentation
        
        if self.transform:
            #ablumentations expects numpy array
            image_np = np.array(image)
            augmented = self.transform(image = image_np)
            image = augmented["image"] 
        
        #get labels
        density_val = row.get('breast_density', 'B') #default to B if missing???
        birads_val = row.get('breast_birads', 1) # default to 1 if missing
        if isinstance(density_val, str) and len(density_val) > 1:
            density_val = density_val[-1] #take last character
        
        
        label_d = torch.tensor(self.density_map.get(density_val, 1), dtype=torch.long)
        label_b = torch.tensor(self.birads_map.get(birads_val, 0), dtype= torch.long)
        return image, label_d, label_b
def config():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument("--csv-file", default=r"C:\Users\louis\Documents\TYP\GhoshData\vindr_detection_v1_folds.csv", type=str)
    parser.add_argument("--img-dir", default=r"C:\Users\louis\Documents\TYP\GhoshData\vindr-mammo-ghosh-png\images_png", type=str)
    parser.add_argument("--output_path", default="./output_swin", type=str)
    
    # Model
    parser.add_argument("--arch", default="swin_tiny_patch4_window7_224", type=str)
    parser.add_argument("--img-size", default=1344, type=int)
    
    # Training
    parser.add_argument("--batch-size", default=2, type=int) # Low batch size for high res!
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num-workers", default=0, type=int) # Set 0 for Windows compatibility
    
    return parser.parse_args()
def main(args):
    seed_all(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Starting Training on: {device}")
    
    #output setup
    os.makedirs(args.out_path, exist_ok= True)
    writer = SummaryWriter(log_dir = os.path.join(args.output_path, "logs"))
    
    #transforms
    train_tfm = load_transform(split="train")
    valid_tfm = load_transform(split = "valid")
    
    