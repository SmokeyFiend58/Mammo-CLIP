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
    
class MammoCLIPDataset(Dataset):
    def __init__(self, dataframe, img_dir, tokenizer, transform_dict = None, split_group = "train", max_len = 128):
        self.data = dataframe.reset_index(drop = True)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.transform_dict = transform_dict
        self.split_group = split_group
        self.max_len = max_len
        self.rare_densities = ['A', 'B', 'D']
        self.density_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        self.birads_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
        self.percent_map = {'A': 0.1, 'B': 0.4, 'C': 0.7, 'D': 0.9}
        
    ### needs completing, dont really know how to implement the scientific annotations/notes ####    
    def prompt_generate(self, row):
            #clean density
        densityUnclean = row.get('breast_density', 'Unknown')
        if isinstance(densityUnclean, str) and "Density" in densityUnclean:
            density = densityUnclean.replace("Density ", "")
        else:
            density = densityUnclean
            #BI-RADS cleaning
            
        biradsUnclean = row.get('breast_birads', 'Unknown')
        if isinstance(biradsUnclean, str) and "BI-RADS" in biradsUnclean:
            birads = biradsUnclean.replace("BI-RADS ", "")
        else: 
            birads = biradsUnclean
            
            #clean findings
        findingString = row.get('finding_categories', "['No Finding']")
        try:
            findings_list = ast.literal_eval(findingString)
            if len(findings_list == 0):
                findingText = "No specific findings"
            else:
                findingsText = ", ".join(findings_list)
        except:
            findingsText = "No findings"
            
        text = f"{row.get('laterality', '')} {row.get('view_position', '')} mammogram. "\
                f"Breast Density {density}. " \
                f"BI-RADS {birads}. "\
                f"Findings: {findingsText}."
        return text
    
    def __len__(self):
            return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
            
            #load image in
        imgID = f"{row['image_id']}.png"
        studyFolder = row['study_id']
        imgPath = os.path.join(self.img_dir, studyFolder, imgID)
        if not os.path.exists(imgPath):
            imgPath = os.path.join(self.img_dirm, imgID)
        
        image = Image.open(imgPath).convert("RGB")
        image_np = np.array(image)
        
        density_val = row.get('breast_density', 'C')
        if isinstance(density_val, str) and len(density_val) > 1:
            densityChar = density_val[-1]
        else: 
            densityChar = 'C'
        
        selected_transform = None
        
        if self.split_group == 'valid':
            selected_transform = self.transform_dict['valid']
        else:
            if densityChar in self.rare_densities:
                selected_transform = self.transform_dict['rare']
            else:
                selected_transform = self.transform_dict['common']
                
        if selected_transform:
            augmented = selected_transform(image = image_np)
            image_tensor = augmented["image"]
        
        text_prompt = self.prompt_generate(row)
        tokens = self.tokenizer(text_prompt, padding = "max_length", truncation = True, max_length = self.max_len, return_tensors= "pt")
        
        
        #squeeze to remove batch dimensions (1,128) goes to 128 
        inputIDs = tokens['input_ids'].squeeze(0)
        attentionMask = tokens['attention_mask'].squeeze(0)
        
        birads_value = row.get('breast_birads',1)
        if isinstance(birads_value, str):
            try: 
                birads_value = int(birads_value.split(' ')[-1])
            except: 
                birads_value = 1
        label_d_class = torch.tensor(self.density_map.get(densityChar, 2), dtype = torch.long)
        label_d_percent = torch.tensor(self.percent_map.get(densityChar, 0.5), dtype= torch.float)
        label_birads = torch.tensor(self.birads_map.get(birads_value, 0), dtype=torch.long)
                
        
        return image_tensor, inputIDs, attentionMask, label_d_class, label_d_percent, label_birads
            
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
    parser.add_argument("--use-center-loss", action="store_true", help="Enable Center Loss")    
    parser.add_argument("--lr-cent", default=0.5, type=float, help="Learning rate for Center Loss")
    parser.add_argument("--cent-weight", default=0.01, type=float, help="Weight for Center Loss")
    
    return parser.parse_args()
def main(args):
    #taken from train_grading with slight modifications
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
    
    
    #optimizer and loss
    #optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    clip_loss_fn = CLIPLoss()
    
    loss_d_class = OrdinalRegressionLoss(num_classes=4)
    loss_d_percent = GaussianUncertaintyLoss()
    loss_birads = OrdinalRegressionLoss(num_classes=5)
    
    #centre loss needs raw visual dim
    visual_dim = getattr(model.visual, "outDim", getattr(model.visual, "out_dim", 768))
    loss_centre = CentreLoss(num_classes = 4, feat_dim = visual_dim, device = device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    #need a higher lr for centre loss
    optimizer_centre = torch.optim.SGD(loss_centre.parameters(), lr = args.lr_cent)
    
    
    #criteria_d = nn.CrossEntropyLoss()
    #criteria_b = nn.CrossEntropyLoss()
    
    #training loop
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc="Training")
        
        for image, input_ids, attention_mask, label_d_class, label_d_percent, label_birads in loop:
            image, input_ids, attention_mask = images.to(device), input_ids.to(device), attention_mask.to(device)
            label_d_class, label_d_percent, label_birads = label_d_class.to(device), label_d_percent.to(device), label_birads.to(device)
            
            optimizer.zero_grad()
            if args.use_centre_loss: 
                optimizer.zero_grad()
            
            #####logits_d, logits_b = model(images)
            
            if args.use_aux_heads:
                out = model(image, {'input_ids': input_ids, 'attention_mask': attention_mask})
                img_emb, text_emb, logit_scale = out[0], out[1], out[2]
                raw_feats = out[3]
                d_class, mu, log_var, b_out = out[4], out[5], out[6], out[7]
            else:
                #standard clip return, above is for the multi head stufs
                img_emb, text_emb, logit_scale = model(image, {'input_ids': input_ids, 'attention_mask': attention_mask})

                
            #forward pass: returns image embeds, text embeds and logit scale
            #img_emb, text_emb, logit_scale = model(image, {'input_ids': input_ids, 'attention_mask': attention_mask})
            
            
            
            #calc loss (calc short for calculate)
            loss = clip_loss_fn(img_emb, text_emb, logit_scale)
            
            if args.use_aux_heads:
                loss += loss_d_class(d_class, label_d_class)
                loss += loss_d_percent(mu, log_var, label_d_percent)
                loss += loss_birads(b_out, label_birads)
                
                if args.use_centre_loss:
                    loss += args.cent_weight * loss_centre(raw_feats, label_d_class)
                
            
            ######loss_d = criteria_d(logits_d, labels_d)
            #######loss_b = criteria_b(logits_b, labels_b)
            
            #######loss = loss_b +loss_d
            
            #back pass
            loss.backward()
            optimizer.step()
            
            #update centre
            if args.use_aux_heads and args.use_centre_loss:
                for param in loss_centre.parameters():
                    param.grad.data *= (1. / args.cent_weight)
                optimizer_centre.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        
        model.eval()
        
        val_loss = 0 
        correct_d = 0
        correct_b = 0
        total_samples = 0
        
        ### progress bar for knowing if the validation is loading
        val_loop = tqdm(valid_loader, desc = "Validating", leave = False)
    
        
        with torch.no_grad():
            for images, input_ids, attention_mask, label_d_class, label_d_percent, label_birads in valid_loader:
                images = images.to(device)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                label_d_class = label_d_class.to(device)
                label_birads = label_birads.to(device)
                
                if args.use_aux_heads:
                    
                    img_emb, text_emb, logit_scale, d_logits, b_logits = model(image, {'input_ids': input_ids, 'attention_mask': attention_mask})
                else:
                    img_emb, text_emb, logit_scale = model(image, {'input_ids': input_ids, 'attention_mask': attention_mask})

                loss = clip_loss_fn(img_emb, text_emb, logit_scale)

                ####loss_d = criteria_d(logits_d, labels_d)
                ####loss_b = criteria_b(logits_b, labels_b)
                
                val_loss += loss.item()
                
                #accuracy for latent sdpace partitioning
                if args.use_aux_heads:
                    #decode ordinal logic
                    pred_d = (d_logits > 0).sum(dim=1)
                    pred_b = (b_logits > 0).sum(dim=1)
                    
                    correct_d += (pred_d == label_d_class).sum().item()
                    correct_b += (pred_b == label_birads).sum.item()
                    total_samples += label_d_class.size(0)
                    
                
        avg_val_loss = val_loss / len(valid_loader)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        
        print(f"Avg Loss: {avg_train_loss:.4f}..... Avg Validation Loss: {avg_val_loss:.4f}")
        
        if args.use_aux_heads and total_samples>0:
            acc_d = correct_d / total_samples
            acc_b = correct_b / total_samples
            print(f"Density Accuracy {acc_d:4f}..... BIRADS Accuracy: {acc_b:.4f}")
            writer.add_scalar("Accuracy/Density", acc_d, epoch)
            writer.add_scaler("Accuracy/BIRADS", acc_b, epoch)
            
        
        #save every epoch to not lose progess
        torch.save(model.state_dict(), os.path.join(args.output_path, f"MammoCLIP_epoch_{epoch+1}.pth"))
    print("Training Complete")
    writer.close()
    
if __name__ == "__main__":
    args = config()
    main(args) 
    
        