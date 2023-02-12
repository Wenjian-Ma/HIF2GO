from torch import nn
import torch
from torch.nn import functional as F


#Focal loss
class BCEFocalLosswithLogits(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLosswithLogits, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: [N, H, W], target: [N, H, W]
        # logits = F.sigmoid(logits)
        alpha = self.alpha
        gamma = self.gamma
        loss = - alpha * (1 - logits) ** gamma * target * torch.log(logits) - \
               (1 - alpha) * logits ** gamma * (1 - target) * torch.log(1 - logits)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
    
#SCE loss
def sce_loss(x, y, alpha=None):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


class sce_kl_loss(nn.Module):#(x, y, alpha=3):
    def __init__(self,num_nodes, model,alpha=3):
        super(sce_kl_loss, self).__init__()
        self.alpha = alpha
        self.num_nodes = num_nodes
        self.model = model
    def forward(self,x,y):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)

        # loss =  - (x * y).sum(dim=-1)
        # loss = (x_h - y_h).norm(dim=1).pow(alpha)

        loss = (1 - (x * y).sum(dim=-1)).pow_(self.alpha)

        loss = loss.mean()

        kl = (0.5 / self.num_nodes) * torch.mean(torch.sum(1 + 2 * self.model.z_log_std_value - torch.pow(self.model.z_mean_value,2) - torch.pow(torch.exp(self.model.z_log_std_value),2), 1))

        loss = loss-kl
        return loss

