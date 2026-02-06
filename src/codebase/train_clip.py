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
from breastclip.model.losses import OrdinalRegressionLoss, GaussianUncertaintyLoss, CentreLoss
from breastclip.data import MammoCLIPDataset
from breastclip.model.training.logger import ResultsLogger
from breastclip.model.training import engine
class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, image_features, text_features, logit_scale):
        # normalisation
        #divide vectors by their length. Turns dot product into cosine similarity
        image_features = image_features / image_features.norm(dim = 1, keepdim = True)
        text_features = text_features / text_features.norm(dim = 1, keepdim = True)
        
        
            
        # similarity matrix (batch x batch)
        # multiply every image vector by every text vector
        # if batch size 8, we get an 8x8 matrix of scores
        # logit scale is a learnable temperature that sharpens predictions
         
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        #t is transpose. flips a matrix over its diagonal
        
        
        #  labels
        # gets correct matches
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size,device = image_features.device)
        
        #loss in two directions
        # given an image did it pick the right text
        # given text did it pick the right image
        loss_i = nn.functional.cross_entropy(logits_per_image, labels)
        loss_t = nn.functional.cross_entropy(logits_per_text, labels)
        
        return (loss_i + loss_t) / 2
    
    #removed MammoCLIPdataset
def config():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument("--csv-file", default=r"C:\Users\louis\Documents\TYP\finding_annotations.csv", type=str)
    parser.add_argument("--img-dir", default=r"C:\Users\louis\Documents\TYP\GhoshData\vindr-mammo-ghosh-png\images_png", type=str)
    parser.add_argument("--output_path", default="./output_clip", type=str)
    
    # Model
    parser.add_argument("--image-encoder", default= "swin_tiny_patch4_window7_224",type=str)
    parser.add_argument("--text-encoder", default="emilyalsentzer/Bio_ClinicalBERT", type=str)
    
    parser.add_argument("--img-size", default=1344, type=int)   
    parser.add_argument("--embed-dim", default=512, type=int)
    

    # Training
    parser.add_argument("--batch-size", default=8, type=int) # Low batch size for high res!
    parser.add_argument("--epochs", default=15, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--val-split", default=0.2, type=float)

    parser.add_argument("--num-workers", default=0, type=int) # Set 0 for Windows compatibility
    #multi head classification stuffs
    
    parser.add_argument("--use-aux-heads", action="store_true", help= "Enabled 3-head architecture")
    parser.add_argument("--use-centre-loss", action="store_true", help="Enable Center Loss")
    parser.add_argument("--use-uncertainty", action="store_true", help="Enable Aleatoric/Epistemic Uncertainty")
    parser.add_argument("--lr-cent", default=0.5, type=float, help="Learning rate for Center Loss")
    parser.add_argument("--cent-weight", default=0.01, type=float, help="Weight for Center Loss")
    
    return parser.parse_args()

def train_one_epoch(model, loader, optimizer, optim_centre, device, args, loss_fns):
    model.train()
    total_loss = 0
    
    for batch in loader:
        img, inp, mask, labelD, labelDp, labelB = batch
        img, inp, mask = img.to(device), inp.to(device), mask.to(device)
        labelD, labelDp, labelB = labelD.to(device), labelDp.to(device), labelB.to(device)
        
        optimizer.zero_grad()
        
        if args.use_centre_loss:
            optim_centre.zero_grad()
        
        #forward pass
        img_emb, text_emb, scale, raw_feats, aux_out = model(img, {'input_ids': inp, 'attention_mask': mask})
        
        #main loss (clip)
        loss = loss_fns['clip'](img_emb, text_emb, scale)
        
        #novelty losses
        if args.use_aux_heads:
            loss += loss_fns['ord_d'](aux_out['d_class'], labelD)
            loss += loss_fns['ord_b'](aux_out['b_class'], labelB)
            
            if args.use_uncertainty:
                loss += loss_fns['gauss'](aux_out['d_perc_mu'], aux_out['d_perc_logvar'], labelDp)
            else:
                #mse is no uncertainty loss
                loss += nn.MSELoss()(aux_out['d_perc_mu'], labelDp.float())

        if args.use_centre_loss:
            loss += 0.01 * loss_fns['centre'](raw_feats, labelD)
        
        
        loss.backward()
        optimizer.step()
        if args.use_centre_loss: 
            optim_centre.step()
        total_loss += loss.item()
        
    return total_loss / len(loader)

                    
    

def main(args):
    #taken from train_grading with slight modifications
    seed_all(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Starting Training on: {device}")
    
    logger = ResultsLogger(args.output_path)
    
    
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
    ##########train_dataframe['clean_density'] = train_dataframe['breast_density'].apply(lambda x: x[-1] if isinstance(x, str) and len(x) >1 else 'C')
    
    #######class_counts = train_dataframe['clean_density'].value_counts().sort_index()
    
    #######class_weights = 1.0/ class_counts
    #assign weights
    #####sample_weights = train_dataframe['clean_density'].map(class_weights).values
    
    #####sampler = WeightedRandomSampler(weights=torch.from_numpy(sample_weights).double(), num_samples=len(train_dataframe), replacement=True)
    
    #now actually create the datasets
    
    #NEW
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder)
    tfm_dict = get_density_augmentation(img_size = args.img_size)
    
    #slight change to parameters
    train_ds = MammoCLIPDataset(train_dataframe, args.img_dir, tokenizer = tokenizer,transform_dict= tfm_dict, split_group="train")
    train_loader = DataLoader(train_ds, batch_size = args.batch_size, shuffle= True, num_workers=args.num_workers, drop_last=True)
    
    valid_ds = MammoCLIPDataset(validation_dataframe, args.img_dir,tokenizer=tokenizer, transform_dict= tfm_dict, split_group="valid")
    valid_loader = DataLoader(valid_ds, batch_size= args.batch_size, shuffle=False, num_workers= args.num_workers, drop_last=True)
    
    

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
    model = MammoCLIP(image_encoder_name= args.image_encoder, text_encoder_name=args.text_encoder, img_size=args.img_size, embed_dim=args.embed_dim, use_aux_heads= args.use_aux_heads).to(device)
    
    #Loss setup to fit with logger
    loss_fns = {'clip': CLIPLoss()}
    if args.use_aux_heads:
        loss_fns['ord_d'] = OrdinalRegressionLoss(num_classes=4)
        loss_fns['ord_b'] = OrdinalRegressionLoss(num_classes=5)
        loss_fns['gauss'] = GaussianUncertaintyLoss()
        
    if args.use_centre_loss:
        visual_dim = getattr(model.visual, "outDim", getattr(model.visual, "out_dim", 768))
        loss_fns['centre'] = CentreLoss(num_classes = 4, feat_dim = visual_dim, device = device)
        optim_centre = torch.optim.SGD(loss_fns['centre'].parameters(), lr = args.lr_cent)
    else:
        optim_centre = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    
    for epoch in range(args.epochs):
        
        train_loss = train_one_epoch(model, train_loader, optimizer, optim_centre, device, args, loss_fns)
        
        print(f"Epoch {epoch}: Train Loss {train_loss: .5f}")
        
        logger.log_epoch(epoch, {'train_loss': train_loss})
        
    logger.saveFinalResult(args, {'final_train_loss': train_loss})
    
        
        
    
if __name__ == "__main__":
    args = config()
    main(args) 
    
        