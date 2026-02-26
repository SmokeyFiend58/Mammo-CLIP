#new file for training on density/ BI-RADS grading. Just image only
import warnings
import argparse
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from utils import seed_all
from breastclip.model.modules.image_encoder import SwinTransformer_Mammo

from breastclip.data.data_utils import load_transform
from breastclip.data.data_utils import get_density_augmentation
from breastclip.model.losses import OrdinalRegressionLoss, DensityMSELoss

warnings.filterwarnings("ignore")

class MultiHeadSwin(nn.Module):
    def __init__(self, encoder_name, img_size, density_loss_type = 'ce', birads_loss_type = 'ce'):
        super().__init__()

    
    #load swin backbone
    
        self.encoder = SwinTransformer_Mammo(
            name = encoder_name,
            pretrained= True,
            img_size= img_size
        )
        inputDim = getattr(self.encoder, "outDim", getattr(self.encoder, "out_dim", 768))
                                        #could be out_dim i cant remember
                                        
                                        
                                        
        
        #head 1 density
        if density_loss_type == 'mse':
            self.head_density = nn.Linear(inputDim, 1)
        else:
            self.head_density = nn.Linear(inputDim, 4)
        
       
        
        # head 2 birads
        if birads_loss_type == 'ordinal':
            self.head_birads = nn.Linear(inputDim, 4)
        else:
            self.head_birads = nn.Linear(inputDim, 5)
            
        
    def forward(self, x):
        features = self.encoder(x)
        dOut = self.head_density(features)
        bOut = self.head_birads(features)
        
        return dOut, bOut
    
#dataset handler
class VinDrSwinDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform_dict = None,split_group = "training"):
        self.data = dataframe.reset_index(drop= True)
        self.img_dir = img_dir
        self.transform_dict = transform_dict
        self.split_group = split_group
        
        self.rare_density = ['A', 'B', 'D']
        
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
        image_np = np.array(image)
        #apply augmentation
        
        density_val = row.get('breast_density', 'C')
        if isinstance(density_val, str) and len(density_val)>1:
            density_val = density_val[-1]
        else:
            density_val = 'C'
        
        selected_transform = None
        
        if self.split_group == 'valid':
            selected_transform = self.transform_dict['valid']
        else:
            if density_val in self.rare_density:
                selected_transform = self.transform_dict['rare']
            else:
                selected_transform = self.transform_dict['common']
        
        if selected_transform:
            augmented = selected_transform(image= image_np)
            image = augmented["image"]
        
        birads_val = row.get('breast_birads', 1)
        if isinstance(birads_val, str):
            try:
                birads_val = int(birads_val.split(' ')[-1])
            except ValueError:
                birads_val = 1
        
        label_d = torch.tensor(self.density_map.get(density_val, 1), dtype=torch.long)
        label_b = torch.tensor(self.birads_map.get(birads_val, 0), dtype= torch.long)
    
        return image, label_d, label_b
    
def config():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument("--csv-file", default="/mnt/nfs/homes/robsonl1/Mammo-CLIP/Mammo-CLIP/DATAFILES/finding_annotations.csv", type=str)
    parser.add_argument("--img-dir", default="/mnt/nfs/homes/robsonl1/Mammo-CLIP/Mammo-CLIP/DATAFILES/GhoshData/vindr-mammo-ghosh-png/images_png", type=str)
    parser.add_argument("--output_path", default="./output_swin", type=str)
    
    # Model
    parser.add_argument("--arch", default="swin_tiny_patch4_window7_224", type=str)
    parser.add_argument("--img-size", default=1344, type=int)
    
    # Training
    parser.add_argument("--batch-size", default=6, type=int) # Low batch size for high res!
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--val-split", default=0.2, type=float)

    parser.add_argument("--num-workers", default=0, type=int) # Set 0 for Windows compatibility
    
    
    parser.add_argument("--density-loss", default= "ce", choices=["ce", "mse"], help="Loss for density")
    parser.add_argument("--birads-loss", default="ce", choices=["ce", "ordinal"], help="Loss for BIRADS")
    
    
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
    
    #clean labels for weight calculation
    train_dataframe['clean_density'] = train_dataframe['breast_density'].apply(lambda x: x[-1] if isinstance(x, str) and len(x) >1 else 'C')
    
    class_counts = train_dataframe['clean_density'].value_counts().sort_index()
    
    class_weights = 1.0/ class_counts
    #assign weights
    sample_weights = train_dataframe['clean_density'].map(class_weights).values
    
    sampler = WeightedRandomSampler(weights=torch.from_numpy(sample_weights).double(), num_samples=len(train_dataframe), replacement=True)
    
    #now actually create the datasets
    tfm_dict = get_density_augmentation(img_size = args.img_size)
    train_ds = VinDrSwinDataset(train_dataframe, args.img_dir, transform_dict= tfm_dict, split_group="train")
    train_loader = DataLoader(train_ds, batch_size = args.batch_size, sampler=sampler,shuffle= False, num_workers=args.num_workers)
    
    valid_ds = VinDrSwinDataset(validation_dataframe, args.img_dir, transform_dict= tfm_dict, split_group="valid")
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
    model = MultiHeadSwin(args.arch, args.img_size, density_loss_type=args.density_loss, birads_loss_type=args.birads_loss).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    if args.density_loss == 'mse':
        criteria_d = DensityMSELoss()
    else: 
        criteria_d = nn.CrossEntropyLoss()
    if args.birads_loss == 'ordinal':
        criteria_b = OrdinalRegressionLoss()
    else:
        criteria_b = nn.CrossEntropyLoss()
    
    #optimizer and loss
    ##criteria_d = nn.CrossEntropyLoss()
    ###criteria_b = nn.CrossEntropyLoss()
    
    #training loop
    scaler = torch.amp.GradScaler('cuda')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc="Training")
        
        for images, labels_d, labels_b in loop:
            images, labels_d, labels_b = images.to(device), labels_d.to(device), labels_b.to(device)
            
            optimizer.zero_grad()
            with torch.autocast(device_type = 'cuda', dtype=torch.float16):
                logits_d, logits_b = model(images)
            
            #calc loss (calc short for calculate)
            
                loss_d = criteria_d(logits_d, labels_d)
                loss_b = criteria_b(logits_b, labels_b)
            
                loss = loss_b +loss_d
            
            #back pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            #loss.backward()
            #optimizer.step()
            scaler.update()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        
        model.eval()
        
        val_loss = 0
        correct_d = 0
        correct_b = 0
        total_samples = 0
        
        
        with torch.no_grad():
            for images, labels_d, labels_b in valid_loader:
                images = images.to(device)
                labels_d = labels_d.to(device)
                labels_b = labels_b.to(device)
                
                logits_d, logits_b = model(images)
                
                loss_d = criteria_d(logits_d, labels_d)
                loss_b = criteria_b(logits_b, labels_b)
                
                val_loss += (loss_d + loss_b).item()
                
                ##decode predictions
                if args.density_loss == 'mse':
                    preds_d = torch.round(logits_d).squeeze().clamp(0, 3).long()
                else: 
                    #arg max
                    preds_d = torch.argmax(logits_d, dim = 1)
                
                #birads decoding
                if args.birads_loss == 'ordinal':
                    #count how many thresholds are passed
                    preds_b = (logits_b > 0).sum(dim=1)
                else:
                    preds_b = torch.argmax(logits_b, dim=1)
                correct_d += (preds_d == labels_d).sum().item()
                correct_b += (preds_b == labels_b).sum().item()
                
                total_samples += labels_d.size(0)
                
                                
        avg_val_loss = val_loss / len(valid_loader)
        acc_d = correct_d / total_samples
        acc_b = correct_b / total_samples
        
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Density", acc_d, epoch)
        writer.add_scalar("Accuracy/BIRADS", acc_b, epoch)
        
        
        print(f"Avg Loss: {avg_train_loss:.4f}")
        print(f"Density Acc: {acc_d:.4f} |||| BI-RADS Acc: {acc_b:.4f}")

        
        #save every epoch to not lose progess
        torch.save(model.state_dict(), os.path.join(args.output_path, f"Swin_epoch_{epoch+1}.pth"))
    print("Training Complete")
    writer.close()
    
if __name__ == "__main__":
    args = config()
    main(args) 
    