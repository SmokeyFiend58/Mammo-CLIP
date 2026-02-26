import torch
import torch.nn as nn
import timm

class SwinEncoder(nn.Module):
    def __init__(self, model_name = 'swin_tiny_patch4_window7_224', pretrained = True, img_size = 1344):
        super().__init__()
        
        #load swin from timm
        #global pool ensures a single vector
        self.backbone = timm.create_model
        (
            model_name,
            pretrained=pretrained
            num_classes= 0 #No classification head yet
            img_size= img_size #1344 is the safe resolution for Swin
            global_pool = 'avg'
        )
        
        #automatically get the embedding dimensions 
        self.embed_dim = self.backbone.num_features
    def forward(self, x):
        #returns raw embeddings (Batch_size, 768)
        return self.backbone(x)
    