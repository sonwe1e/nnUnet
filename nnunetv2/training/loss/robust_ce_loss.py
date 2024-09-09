import torch
from torch import nn, Tensor
import numpy as np
import torch.nn.functional as F


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())


class RobustFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        """
        Initializes the RobustFocalLoss module.

        Args:
        alpha (float): Balancing factor, default is 1.
        gamma (float): Focusing parameter, default is 2.
        reduction (str): Reduction method to apply to the output ('none', 'mean', or 'sum').
        """
        super(RobustFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Computes the Focal Loss for classification tasks.

        Args:
        input (torch.Tensor): Predicted logits tensor of shape (batch_size, num_classes, ...).
        target (torch.Tensor): Ground truth tensor of shape (batch_size, ...) with values in the range [0, num_classes-1].

        Returns:
        torch.Tensor: Focal loss.
        """
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        target = target.long()

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target.unsqueeze(1))
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -self.alpha * (1 - pt) ** self.gamma * logpt
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class TopKLoss(RobustCrossEntropyLoss):
    """
    input must be logits, not probabilities!
    """

    def __init__(
        self,
        weight=None,
        ignore_index: int = -100,
        k: float = 10,
        label_smoothing: float = 0,
    ):
        self.k = k
        super(TopKLoss, self).__init__(
            weight, False, ignore_index, reduce=False, label_smoothing=label_smoothing
        )

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(
            res.view((-1,)), int(num_voxels * self.k / 100), sorted=False
        )
        return res.mean()
