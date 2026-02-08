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
class GaussianUncertaintyLoss(nn.Module):
    #regression loss for density percent
    def __init__(self):
        super().__init__()
        self.nll = nn.GaussianNLLLoss()
    def forward(self, predMean, pred_log_var, target):
        #pred log.. comes from the network
        var = torch.exp(pred_log_var)
        #ensuring variance is positive
        
        if target.dim() == 1:
            target = target.view(-1,1)
        return self.nll(predMean, target, var)
    
class DensityMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self, logits, targets):
        targetsFloat = targets.float().view(-1,1)
        return self.mse(logits, targetsFloat)
    
class CentreLoss(nn.Module):
    # penalties on the distance between latent features and class centres
    def __init__(self, num_classes, feat_dim, device = 'cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        
        #lernable parameter(different from the model weights) 
        self.centres = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))
        
    def forward(self, features, labels):
        batch_size = features.size(0)
        
        centres_batch = self.centres.index_select(0, labels.long())
        
        loss = (features - centres_batch).pow(2).sum() / 2.0 / batch_size
        return loss
    
    
    
    