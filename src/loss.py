import torch
import torch.nn as nn

## Supervied contrastive loss
def supCL_loss(criterion, anchor, pos, neg, new_w=0, temper=0.05):
    cos = nn.CosineSimilarity(dim=-1)
    pos_sim = cos(anchor.unsqueeze(1), pos.unsqueeze(0)) / temper
    neg_sim = cos(anchor.unsqueeze(1), neg.unsqueeze(0)) / temper

    cos_sim = torch.cat([pos_sim, neg_sim], dim=1)
    labels = torch.arange(cos_sim.size(0)).long().to(anchor.device)
    weights = torch.tensor(
        [
            [0.0] * (cos_sim.size(-1) - neg_sim.size(-1))
            + [0.0] * i
            + [new_w]
            + [0.0] * (neg_sim.size(-1) - i - 1)
            for i in range(neg_sim.size(-1))
        ]
    ).to(anchor.device)
    cos_sim += weights
    loss = criterion(cos_sim, labels)
    return loss
