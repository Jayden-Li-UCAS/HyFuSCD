import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight=weight, ignore_index=ignore_index,
                                   reduction='elementwise_mean')

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


class ChangeSimilarity(nn.Module):
    def __init__(self, reduction='mean'):
        super(ChangeSimilarity, self).__init__()
        self.loss_f = nn.CosineEmbeddingLoss(margin=0.1, reduction=reduction)

    def forward(self, x1, x2, label_change):
        b, c, h, w = x1.size()
        x1 = F.softmax(x1, dim=1)
        x2 = F.softmax(x2, dim=1)
        x1 = x1.permute(0, 2, 3, 1)
        x2 = x2.permute(0, 2, 3, 1)
        x1 = torch.reshape(x1, [b * h * w, c])
        x2 = torch.reshape(x2, [b * h * w, c])

        label_unchange = ~label_change.bool()
        target = label_unchange.float()
        target = target - label_change.float()
        target = torch.reshape(target, [b * h * w])

        loss = self.loss_f(x1, x2, target)
        return loss

class BCEWithIgnoreLoss(nn.Module):
    def __init__(self, ignore_index=255, OHEM=False):
        super().__init__()
        self.ignore_index = ignore_index
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.OHEM = OHEM

    def forward(self, logits, target):
        if len(logits.shape) != len(target.shape) and logits.shape[1] == 1:
            logits = logits.squeeze(1)

        target = target.float()
        valid_mask = (target != self.ignore_index)
        loss = self.bce(logits, target)

        # OHEM
        if self.OHEM:
            loss_, _ = loss.contiguous().view(-1).sort()
            min_value = loss_[int(0.5 * loss.numel())]

            loss = loss[valid_mask]
            loss = loss[loss >= min_value]
        else:
            loss = loss[valid_mask]

        return loss.mean()


class BCDLoss(nn.Module):
    def __init__(self,
                 losses=[BCEWithIgnoreLoss(), DiceLoss(mode='binary', ignore_index=255)],
                 loss_weight=[1, 1]):
        super(BCDLoss, self).__init__()
        self.loss_weights = loss_weight
        self.losses = losses

    def forward(self, logits, target):
        losses = {}
        for i in range(len(self.losses)):
            loss = self.losses[i](logits, target)
            losses[i] = loss * self.loss_weights[i]
        losses["loss"] = sum(losses.values())
        return losses["loss"]
