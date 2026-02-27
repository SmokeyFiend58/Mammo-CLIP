# this is the test loop and should calc various difference accuracies aswell

import logging
import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from src.codebase.breastclip.model.mammo_clip import MammoCLIP
from src.codebase.train_grading import MultiHeadSwin


log = logging.getLogger(__name__)

class MammoEval:
    def __init__(self, model, dataloader, device, output_path):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.out_path = output_path
        
        os.makedirs(self.out_path, exist_ok = True)
    
    def decodeOrdinal(self, logits):
        #converts ordinal logits(batch, num_classes-1) into class labels
        # sigmoid restricts logits to 0-1 range (prob of crossing threshold into another classs)
        probs = torch.sigmoid(logits)
        #sum number of thresholds gives the class index (0 - n)
        pred_labels = (probs > 0.5).sum(dim=1)
        
        return pred_labels, probs

    def calcECE(self, preds, labels, n_bins = 10):
        #expected calibration error
        # checks if a 90% confidence is a 90% accuracy
        bin_boundaries = torch.linspace(0,1,n_bins +1)
        ece = 0.0
        
    ########        #if preds are logics/class indices we need confidence scores
    ####assuming preds passed here are the max probabilities for the chosen class    
        confidences = preds
        accuracies = (preds  == labels)# placeholder needs raw probs 
    ####correct loigic in the main loop handles preparation
        return 0.0
    
    def enable_Dropout(self, m):
        if type(m) == torch.nn.Dropout:
            m.train()
    def evalMetrics(self):
        self.model.eval()
        
        all_labels_density = [] #ground trush labels 
        all_labels_birads = []
        
        all_preds_density = [] #class predictions
        all_preds_birads = []
        
        all_probs_density = [] #density probabilities
        all_aleatoric = [] # density % variance
        
        print("Running eval")
        with torch.no_grad():
            for batch in tqdm(self.dataloader):
                #unpack batch 
                #adjust batch unpacking depending if loader returns a dict or tuple
                if isinstance(batch, dict):
                    img = batch['image'].to(self.device)
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels_density = batch['labels_d'].to(self.device)
                    labels_birads = batch['labels_b'].to(self.device)
                    text_inputs = {'input_ids': input_ids, 'attention_mask':attention_mask}
                else:
                    #if using train_grading as this returns typle
                    img, labels_density, labels_birads = batch
                    img, labels_density, labels_birads = img.to(self.device), labels_density.to(self.device), labels_birads.to(self.device)
                    input_ids = None # slight change to make it work with train_grading
                
                try: 
                    #attempt to do VLM
                    if isinstance(self.model, MammoCLIP):
                        
                        
                    
                #forward pass
                #need the aux out dict from mammo_clip

                        _,_,_,_,aux_out = self.model(img, {'input_ids': input_ids, 'attention_mask':attention_mask})
                        d_logits = aux_out['d_class']
                        b_logits = aux_out['b_class']
                        
                        if 'd_percent_logvar' in aux_out and aux_out['d_percent_logvar'] is not None:
                            aleatoric = torch.exp(aux_out['d_percent_logvar'])
                            all_aleatoric.extend(aleatoric.cpu().numpy())
                    else:
                        #fallback for image only
                        d_logits, b_logits = self.model(img)
                except RuntimeError as e:
                    print(f"Forward pass error: {e}")
                    continue
                
                
                
                #density decoding
                #ordinal decoding for density class
                if d_logits.shape[1] > 1:
                    pred_density_class, d_probs_raw = self.decodeOrdinal(d_logits)
                    all_probs_density.extend(torch.sigmoid(d_logits).mean(dim=1).cpu().numpy())
                else:
                    #regression MSE case 
                    pred_density_class = torch.round(d_logits).clamp(0,3)
                    all_probs_density.extend(torch.sigmoid(d_logits).mean(dim=1).cpu().numpy())

                #Birads decoding
                #oridnal deocidng for BIRADS
                if b_logits.shape[1] > 1:
                    pred_birads_class, _ = self.decodeOrdinal(b_logits)
                else:
                    pred_birads_class = torch.round(b_logits).clamp(0, 4)
                
                all_labels_density.extend(labels_density.cpu().numpy())
                all_labels_birads.extend(labels_birads.cpu().numpy())
                
                all_preds_density.extend(pred_density_class.cpu().numpy())
                all_preds_birads.extend(pred_birads_class.cpu().numpy())
                
                
                #aleatoric uncertainty
                #use exp() because model outputs log_variance
                
                    
                
                
                
                #auroc needs standard probability distribution
                #ordinal outputs are thresholds, we approx class prob
                #take mean of the sigmoid ouytputs (simpliified for AUROC)
        
        
        #metric calculation
        
        #marcof1 and accuracy, macro average treats all classes equally
        f1_density = f1_score(all_labels_density, all_preds_density, average='macro')
        acc_density = accuracy_score(all_labels_density, all_preds_density)
        
        f1_birads = f1_score(all_labels_birads, all_preds_birads, average='macro')
        acc_birads = accuracy_score(all_labels_birads, all_preds_birads)
        
        # one vs rest auroc
        try: 
            auroc_density = roc_auc_score(all_labels_density, all_probs_density, multi_class='ovr')
        except:
            auroc_density = 0.0 #fallbackl for if only 1 class present in batch
        
        print(f"Density --- F1 (macro): {f1_density:.4f} --- Accuracy: {acc_density:.4f} --- AUROC: {auroc_density:.4f}\n")
        
        print(f"BIRADS --- F1 (macro): {f1_birads:.4f} --- Accuracy: {acc_birads:.4f}")
        
        #per class sensitivity 
        #confusion matrix
        
        confusion_matrix_density = confusion_matrix(all_labels_density, all_preds_density)
        print("\n Density Confusion Matirx: \n", confusion_matrix_density)
        
        return {"f1_density": f1_density, "f1_birads": f1_birads, "aleatoric": np.mean(all_aleatoric)if all_aleatoric else 0}
    
    def evalUncertaintyMCDROPOUT(self, mc_samples = 10):
        self.model.eval()
        
        #force dropout layers to stay active during eval
        self.model.apply(self.enable_Dropout)
        
        epistemic_uncertainty = []
        
        with torch.no_grad():
            for batch in tqdm(self.dataloader):
                if isinstance(batch, dict):
                    img = batch['image'].to(self.device)
                    #dont need text 
                else:
                    img, _, _ = batch
                    img = img.to(self.device)
                    
                batch_predict = []
                
                #mc loop, predict N times for this batch
                for _ in range(mc_samples):
                    #handle both model types
                
                    output = self.model(img, None) if isinstance(self.model, MammoCLIP) else self.model(img)
                    
                    if isinstance(output, tuple) and len(output) ==5:
                        aux_out = output[4]
                        
                    
                        logits = aux_out["d_class"]
                    elif isinstance(output, tuple) and len(output) ==2:
                        # this is image only
                        logits = output[0]
                        
                        
                    probs = torch.sigmoid(logits) # normalize
                    batch_predict.append(probs.unsqueeze(0)) #shape = 1, batch, classes-1
                    
                # stack: samples, batch, classes-1
                batch_predict = torch.cat(batch_predict, dim=0)
                
                #calc variance across sample dimension (dim = 0)
                #high variances means the model is very uncertain
                variance = batch_predict.var(dim=0).mean(dim=1) # average variance across classes
                epistemic_uncertainty.extend(variance.cpu().numpy())
            average_epistemic = np.mean(epistemic_uncertainty)
            print(f"Average Epistemic Uncertainty: {average_epistemic:.5f}")
            return average_epistemic
        
                    
        
                
                

        

        
        