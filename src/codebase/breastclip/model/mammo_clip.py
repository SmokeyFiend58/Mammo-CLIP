#image and text
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from breastclip.model.modules.image_encoder import SwinTransformer_Mammo
from breastclip.model.modules.text_encoder import HuggingfaceTextEncoder

class MammoCLIP(nn.Module):
    def __init__(self, image_encoder_name = "swin_tiny_patch4_window7_224", text_encoder_name = "emilyalsentzer/Bio_ClinicalBERT", img_size = 1344, embed_dim = 256,):
        super().__init__()
        
        #image branch
        self.visual = SwinTransformer_Mammo(name=image_encoder_name, pretrained=True, img_size=img_size)
        
        visual_dim = self.visual.outDim
        
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
        
    def forward(self, image, text_inputs):
        # image [batch_size, 3, H, W]
        # text inputs: dictionary from tokenizer containing input_ids and attention_mask
        
        # image encoding
        images_features = self.visual(image)
        
        #text encoding 
        # text encoder returns batch, seq length, text_dim
        text_ouputs = self.text_encoder(text_inputs)
        
        #pooling: take the first token CLS (index 0) to represent the sentence
        text_features = text_ouputs[:, 0, :]
        
        # projection
        image_embeds = self.image_projection(images_features)
        text_embeds = self.text_projection(text_features)
        
        #normalisation
        # normalise vectors to length 1
        image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim = True)
        text_embeds = text_embeds / text_embeds.norm(dim = 1, keepdim = True)
        
        return image_embeds, text_embeds, self.logit_scale.exp()
    