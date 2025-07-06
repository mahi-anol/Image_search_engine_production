import torch
import torch.nn as nn
import torch.nn.functional as F

class contrastive_loss(nn.Module):
    def __init__(self,margin=2.0):
        super().__init__()
        self.margin=margin

    def forward(self,sample1_embedding,sample2_embedding,pos_or_neg):
        pos_or_neg=pos_or_neg.float()
        euclidian_distance=F.pairwise_distance(sample1_embedding,sample2_embedding)
        loss_if_similar=pos_or_neg*torch.pow(euclidian_distance,2)
        loss_if_not_similar=(1-pos_or_neg)*torch.pow(torch.clamp(self.margin-euclidian_distance,min=0.0), 2)
        losses=loss_if_similar+loss_if_not_similar
        loss=torch.mean(losses)
        return loss
    
class triplet_loss(nn.Module):
    def __init__(self,margin=2.0):
        super().__init__()
        self.margin=margin
    def forward(self,anchor,positive,negative):
        distance_positive=F.pairwise_distance(anchor,positive)
        distance_negative=F.pairwise_distance(anchor,negative)
        losses=torch.relu(torch.pow(distance_positive,2)-torch.pow(distance_negative,2)+self.margin)
        loss=torch.mean(losses)
        return loss

