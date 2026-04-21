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
from src.codebase.utils import seed_all
from src.codebase.breastclip.data.data_utils import get_density_augmentation
from src.codebase.breastclip.model.mammo_clip import MammoCLIP
from src.codebase.breastclip.model.losses import OrdinalRegressionLoss, GaussianUncertaintyLoss, CentreLoss
from src.codebase.breastclip.data.MammoCLIPDataset import MammoCLIPDataset, clean_density_letter
from src.codebase.breastclip.model.training.logger import ResultsLogger
import json

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
    
def load_synthetic_reports(reportsJSONPath):
    #returns a dict mapping studyid to report text
    
    with open(reportsJSONPath) as f:
        data = json.load(f)
    report_map = {}
    
    for study in data["studies"]:
        #combines findings text+impresion into one string
        findings = " ".join(
            f["text"] for f in study["radiology_report"]["findings"])
        impression = study["radiology_report"]["impression"]["text"]
        report_map[study["study_id"]] = f"{findings} {impression}"
    return report_map

    
    
def config():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument("--csv-file", default="/mnt/nfs/homes/robsonl1/Mammo-CLIP/Mammo-CLIP/DATAFILES/finding_annotations.csv", type=str)
    parser.add_argument("--img-dir", default="/mnt/nfs/homes/robsonl1/Mammo-CLIP/Mammo-CLIP/DATAFILES/GhoshData/vindr-mammo-ghosh-png/images_png", type=str)
    parser.add_argument("--output_path", default="./output_clip", type=str)
    
    # Model
    parser.add_argument("--image-encoder", default= "swinv2_tiny_window8_256",type=str)
    #parser.add_argument("--text-encoder", default="emilyalsentzer/Bio_ClinicalBERT", type=str)
    parser.add_argument("--text-encoder", default="fixed_clinicalbert", type=str)
    #errors as partitions must be divisiable 128
    parser.add_argument("--img-size", default=1280, type=int)   
    parser.add_argument("--embed-dim", default=512, type=int)
    

    # Training
    parser.add_argument("--batch-size", default=2, type=int) # Low batch size for high res!
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
    
    parser.add_argument("--use-synth-reports", action= "store_true", help="Use synthesized radiology reports as text input")
    parser.add_argument("--reports-json", type=str, default=None, help="Path to reports.json from report_synthesizer.py")
    
    
    return parser.parse_args()

def train_one_epoch(model, loader, optimizer, optim_centre, device, args, loss_fns, scalar, epoch):
    model.train()
    total_loss = 0
    #progress bar added
    loop = tqdm(loader, desc=f"Epoch{epoch}", leave=True)
    
    
    for batch in loop:
        img, inp, mask, labelD, labelDp, labelB = batch
        img, inp, mask = img.to(device), inp.to(device), mask.to(device)
        labelD, labelDp, labelB = labelD.to(device), labelDp.to(device), labelB.to(device)
        
        optimizer.zero_grad()
        
        if args.use_centre_loss:
            optim_centre.zero_grad()
        with torch.amp.autocast('cuda'):
        #forward pass
            img_emb, text_emb, scale, raw_feats, aux_out = model(img, {'input_ids': inp, 'attention_mask': mask})
        
        #main loss (clip)
            loss = loss_fns['clip'](img_emb, text_emb, scale)
        
        #novelty losses
            if args.use_aux_heads:
                loss += loss_fns['ord_d'](aux_out['d_class'], labelD)
                loss += loss_fns['ord_b'](aux_out['b_class'], labelB)
            
                if args.use_uncertainty:
                    loss += loss_fns['gauss'](aux_out['d_percent_mu'], aux_out['d_percent_logvar'], labelDp)
                else:
                #mse is no uncertainty loss
                    loss += nn.MSELoss()(aux_out['d_percent_mu'], labelDp.float())

            if args.use_centre_loss:
                loss += args.cent_weight * loss_fns['centre'](raw_feats, labelD)
        
        scalar.scale(loss).backward()
        #loss.backward()
        #optimizer.step()
        scalar.unscale_(optimizer)
        if args.use_centre_loss:
            scalar.unscale_(optim_centre)
        scalar.step(optimizer)
        if args.use_centre_loss:
            scalar.step(optim_centre)
        scalar.update()
        total_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")
        
        
    return total_loss / len(loader)

def val_one_epoch(model, loader,device, args, loss_fns):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc = "Validating", leave = False):
            img, inp, mask, labelD, labelDp, labelB = batch
            img, inp, mask = img.to(device), inp.to(device), mask.to(device)
            labelD, labelDp, labelB = labelD.to(device), labelDp.to(device), labelB.to(device)
        
            with torch.amp.autocast('cuda'):
                img_emb, text_emb, scale, raw_feats, aux_out = model(img, {'input_ids': inp, 'attention_mask': mask})                   

                loss = loss_fns['clip'](img_emb, text_emb, scale)
                
                if args.use_aux_heads:
                    loss += loss_fns['ord_d'](aux_out['d_class'], labelD)
                    loss += loss_fns['ord_b'](aux_out['b_class'], labelB)
            total_loss += loss.item()
    
    averageLoss = total_loss / len(loader)
    
    print(f"Validation Loss: {averageLoss:.5f}")
    
    return averageLoss

        
        

def main(args):
    #taken from train_grading with slight modifications
    seed_all(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Starting Training on: {device}")
    
    logger = ResultsLogger(args.output_path)
    
    
    #load all data
    full_dataFrame = pd.read_csv(args.csv_file)
    
    #if the flag is set, overwrite the text column with synthesized reports
    if args.use_synth_reports:
        assert args.reports_json is not None, \
            "--reports-json required when --use-synth-reports is set"
        report_map = load_synthetic_reports(args.reports_json)
        full_dataFrame["text"] = full_dataFrame["study_id"].map(report_map)
        missing = full_dataFrame["text"].isnull().sum()
        
        if missing > 0:
            print(f"Warning: {missing} rows had no matching report, dropping them")
            full_dataFrame = full_dataFrame.dropna(subset=["text"])
            

    
    #isolate training data
    if 'split' in full_dataFrame.columns:
        
        train_full_dataframe = full_dataFrame[full_dataFrame['split'] == 'training']
    else: 
        train_full_dataframe = full_dataFrame
        print("Warning warning warning: no split column for some stupid reason")
    
    #create train/val split
    #stratified split also
    #train_dataframe, validation_dataframe = train_test_split(train_full_dataframe, test_size= args.val_split, random_state=args.seed, stratify=train_full_dataframe['breast_density'] if 'breast_density' in train_full_dataframe.columns else None)
    
    stratify_column = train_full_dataframe['breast_density'].apply(clean_density_letter) if 'breast_density' in train_full_dataframe.columns else None
    
    train_dataframe, validation_dataframe = train_test_split(train_full_dataframe, test_size = args.val_split, random_state=args.seed, stratify=stratify_column)
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder)
    tfm_dict = get_density_augmentation(img_size = args.img_size, include_flip = False)
    
    #slight change to parameters
    train_ds = MammoCLIPDataset(train_dataframe, args.img_dir, tokenizer = tokenizer,transform_dict= tfm_dict, split_group="train")
    train_loader = DataLoader(train_ds, batch_size = args.batch_size, shuffle= True, num_workers=args.num_workers, drop_last=True)
    
    valid_ds = MammoCLIPDataset(validation_dataframe, args.img_dir,tokenizer=tokenizer, transform_dict= tfm_dict, split_group="valid")
    #dont drop_last in the valid because the validation should see every sample
    valid_loader = DataLoader(valid_ds, batch_size= args.batch_size, shuffle=False, num_workers= args.num_workers, drop_last=False)
    
    

    #model
    model = MammoCLIP(image_encoder_name= args.image_encoder, text_encoder_name=args.text_encoder, img_size=args.img_size, embed_dim=args.embed_dim, use_aux_heads= args.use_aux_heads).to(device)
    for param in model.text_encoder.parameters():
        param.requires_grad = False
        
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

    scalar = torch.amp.GradScaler()
    
    bestValidationLoss = float('inf')
    
    patience = 5
    patienceCounter = 0
    
    
    for epoch in range(args.epochs):
        
        train_loss = train_one_epoch(model, train_loader, optimizer, optim_centre, device, args, loss_fns, scalar, epoch)
        val_loss = val_one_epoch(model, valid_loader, device, args, loss_fns)
        
        
        
        print(f"Epoch {epoch}: Train Loss {train_loss: .5f}. Validation loss {val_loss:.5f}")
        
        
        logger.log_epoch(epoch, {'train_loss': train_loss, 'val_loss': val_loss})
        if val_loss < bestValidationLoss:
            bestValidationLoss = val_loss
            patienceCounter = 0
            
            torch.save(model.state_dict(), os.path.join(args.output_path, "Best_clip_model.pth"))
            print(f"New best validation loss {bestValidationLoss:.5f}")
            
        else:
            patienceCounter += 1
            print(f" No improvement ({patienceCounter}/{patience})")
            if patienceCounter >= patience:
                print(f"Early stopping at epoch {epoch+1}, best validation loss: {bestValidationLoss:.5f}")
                break
            
            
        
        
    logger.saveFinalResult(args, {'final_train_loss': train_loss, 'best_validation_loss': bestValidationLoss})
    
    
    
        
        
    
if __name__ == "__main__":
    args = config()
    main(args) 
    
        