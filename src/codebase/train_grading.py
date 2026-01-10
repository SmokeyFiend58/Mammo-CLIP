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
from sklearn.model_selection import train_test_split
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
    def __init__(self, dataframe, img_dir, split_group = "training", transform = None):
        self.data = dataframe.reset_index(drop= True)
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
        if isinstance(density_val, str) and len(density_val)>1:
            density_val = density_val[-1]
        
        birads_val = row.get('breast_birads', 1) # default to 1 if missing
        if isinstance(birads_val, str):
            try:
                birads_val = int(birads_val.split(' ')[-1])
            except ValueError:
                birads_val = 1 #fallback
                
        
        label_d = torch.tensor(self.density_map.get(density_val, 1), dtype=torch.long)
        label_b = torch.tensor(self.birads_map.get(birads_val, 0), dtype= torch.long)
        return image, label_d, label_b
def config():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument("--csv-file", default=r"C:\Users\louis\Documents\TYP\finding_annotations.csv", type=str)
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
    parser.add_argument("--val-split", default=0.2, type=float)

    parser.add_argument("--num-workers", default=0, type=int) # Set 0 for Windows compatibility
    
    return parser.parse_args()
def main(args):
    seed_all(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Starting Training on: {device}")
    
    #output setup
    os.makedirs(args.output_path, exist_ok= True)
    writer = SummaryWriter(log_dir = os.path.join(args.output_path, "logs"))
    
    #load all data
    full_dataFrame = pd.read_csv(args.csv_file)
    
    #isolate training data
    if 'split' in full_dataFrame.columns:
        
        train_full_dataframe = full_dataFrame[full_dataFrame['split'] == 'training']
    else: 
        train_full_dataframe = full_dataFrame
        print("Warning warning warning: no split column for some stupid reason")
    
    #create train/val split
    #stratified split also
    train_dataframe, validation_dataframe = train_test_split(train_full_dataframe, test_size= args.val_split, random_state=args.seed, stratify=train_full_dataframe['breast_density'] if 'breast_density' in train_full_dataframe.columns else None)
    # it says test size in the function but the only data available to get is from the training split so dont worry
    
    #now actually create the datasets
    
    train_tfm = load_transform(split = "train")
    train_ds = VinDrSwinDataset(train_dataframe, args.img_dir, transform= train_tfm)
    train_loader = DataLoader(train_ds, batch_size = args.batch_size, shuffle= True)
    
    # validation (no augmentation, just resize)
    valid_tfm = load_transform(split = "valid")
    valid_ds = VinDrSwinDataset(validation_dataframe, args.img_dir, transform= valid_tfm)
    valid_loader = DataLoader(valid_ds, batch_size= args.batch_size, shuffle=False)
    
    """#transforms
    train_tfm = load_transform(split="train")
    valid_tfm = load_transform(split = "valid")
    
    #datasets
    #in real thing, split the csv's into the splits already predefineed
    #for now, load the same csv for both to get something running
    
    train_ds = VinDrSwinDataset(args.csv_file, args.img_dir, transform=train_tfm)
    train_loader = DataLoader(train_ds, batch_size= args.batch_size, shuffle=True)
    
    print(f"Dataset loaded{len(train_ds)}")
    """
    
    #model
    model = MultiHeadSwin(args.arch, args.img_size).to(device)
    
    #optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criteria_d = nn.CrossEntropyLoss()
    criteria_b = nn.CrossEntropyLoss()
    
    #training loop
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc="Training")
        
        for images, labels_d, labels_b in loop:
            images, labels_d, labels_b = images.to(device), labels_d.to(device), labels_b.to(device)
            
            optimizer.zero_grad()
            
            logits_d, logits_b = model(images)
            
            #calc loss (calc short for calculate)
            
            loss_d = criteria_d(logits_d, labels_d)
            loss_b = criteria_b(logits_b, labels_b)
            
            loss = loss_b +loss_d
            
            #back pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        
        model.eval()
        
        val_loss = 0 
        
        with torch.no_grad():
            for images, labels_d, labels_b in valid_loader:
                images = images.to(device)
                labels_d = labels_d.to(device)
                labels_b = labels_b.to(device)
                
                logits_d, logits_b = model(images)
                
                loss_d = criteria_d(logits_d, labels_d)
                loss_b = criteria_b(logits_b, labels_b)
                
                val_loss += (loss_d + loss_b).item()
        avg_val_loss = val_loss / len(valid_loader)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        
        print(f"Avg Loss: {avg_train_loss:.4f}")
        
        #save every epoch to not lose progess
        torch.save(model.state_dict(), os.path.join(args.output_path, f"Swin_epoch_{epoch+1}.pth"))
    print("Training Complete")
    writer.close()
    
if __name__ == "__main__":
    args = config()
    main(args) 
    