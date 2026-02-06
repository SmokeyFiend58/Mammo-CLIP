#image and text
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from breastclip.model.modules.image_encoder import SwinTransformer_Mammo
from breastclip.model.modules.text_encoder import HuggingfaceTextEncoder

class MammoCLIP(nn.Module):
    def __init__(self, image_encoder_name = "swin_tiny_patch4_window7_224", text_encoder_name = "emilyalsentzer/Bio_ClinicalBERT", img_size = 1344, embed_dim = 256,use_aux_heads = False, use_uncertainty = False): # aux heads are for activiating a novelty
        super().__init__()
        self.use_aux_heads = use_aux_heads
        self.use_uncertainty = use_uncertainty
        
        
        
        #image branch
        self.visual = SwinTransformer_Mammo(name=image_encoder_name, pretrained=True, img_size=img_size)
        
#        visual_dim = self.visual.outDim ...... changed to be a bit more robust
        visual_dim = getattr(self.visual, "outDim", getattr(self.visual, "out_dim", 768)) 
               
        #text branch
        self.text_encoder = HuggingfaceTextEncoder(name = text_encoder_name, pretrained=True)
        
        text_dim = self.text_encoder.out_dim
        
        #BERT and swin output different sizes, need to make them the same
        
        #projections(mapping in shared space)
        self.image_projection = nn.Linear(visual_dim, embed_dim, bias=False)
        self.text_projection = nn.Linear(text_dim, embed_dim, bias = False)
        
        # temperature parameter
        # CLIP uses a learnable temperature to scale logits
        # Initialize it to log(1/0.07) like the Mammo-CLIP paper
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))
        
        #3 heads on the image tower
        if self.use_aux_heads:
            #Head A: density class (ordinal: 4 classes -> 3 thresholds)
            self.head_density_class = nn.Linear(visual_dim, 3)
            
            d_perc_out = 2 if self.use_uncertainty else 1
            
            
            #head b: density % (regression: mean and logVar)
            self.head_density_perc = nn.Linear(visual_dim, d_perc_out)
            
            #head c: BIRADS (ordinal: 5 classes -> 4 thresholds)
            self.head_birads = nn.Linear(visual_dim, 4)
            
            #dropout for monte carlo sampling .......... epistemic uncertainty
            self.dropout = nn.Dropout(p=0.2)
            
            
        
        
        
    def forward(self, image, text_inputs):
        # image [batch_size, 3, H, W]
        # text inputs: dictionary from tokenizer containing input_ids and attention_mask
        
        # image encoding
        images_features = self.visual(image)
        
        #text encoding 
        # text encoder returns batch, seq length, text_dim
        image_embeds = self.image_projection(images_features)
        image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim = True)

        
        text_embeds = None
        if text_inputs is not None:
            
            text_ouputs = self.text_encoder(text_inputs)
            #pooling: take the first token CLS (index 0) to represent the sentence
            text_features = text_ouputs[:, 0, :]
            # projection
            text_embeds = self.text_projection(text_features)
           
            #normalisation
            # normalise vectors to length 1
            text_embeds = text_embeds / text_embeds.norm(dim = 1, keepdim = True)
        
        
        
     
        aux_out = {}
        #auxlirary outputs!!
        if self.use_aux_heads:
            #applying dropout only if training or doing mc sampling
            features_dropped = self.dropout(images_features) if self.use_uncertainty else images_features
            aux_out['d_class'] = self.head_density_class(features_dropped)
            aux_out['b_class'] = self.head_birads(features_dropped)
            
            d_percent_raw = self.head_density_perc(features_dropped)
            if self.use_uncertainty:
                aux_out['d_percent_mu'] = d_percent_raw[:, 0]
                aux_out['d_percent_logvar'] = d_percent_raw[:, 1]
            else: 
                aux_out['d_percent_mu'] = d_percent_raw.squeeze()
                aux_out['d_percent_logvar'] = None
            
            
            #return everything
        return image_embeds, text_embeds, self.logit_scale.exp(), images_features, aux_out
        
    