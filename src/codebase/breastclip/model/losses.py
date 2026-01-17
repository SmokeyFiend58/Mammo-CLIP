import torch
import torch.nn as nn
class OrdinalRegressionLoss(nn.Module):
    def __init__(self, num_classes = 5):
        super().__init__()
        self.num_classes = num_classes
        
        #use BCEwithlogits because each question for ordinal loss is binary
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    def forward(self, logits, targets):
        #logit shape : batch,num_classes -1 ===== [8,4]
        #target shape: batch ===== [8] containing indiicies 0-4
        
        batchSize = logits.shape[0]
        
        #questions grid
        labelSpace = torch.arange(self.num_classes -1, device= logits.device).expand(batchSize-1)
        
        #expand targets
        targetsExpanded = targets.unsqueeze(1).expand(batchSize, self.num_classes -1)
        
        #create binary targets: 1 if target >label index
        ordinalTargets = (targetsExpanded > labelSpace).float()
        
        loss = self.bce(logits, ordinalTargets)
        return loss.sum(dim=1).mean()

class DensityMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self, logits, targets):
        targetsFloat = targets.float().view(-1,1)
        return self.mse(logits, targetsFloat)
    
    
    