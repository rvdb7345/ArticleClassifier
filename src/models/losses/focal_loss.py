"""This file defines the FocalLoss function."""
import torch

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super().__init__()
        self._gamma = gamma
        self._alpha = alpha

    def forward(self, y_pred, y_true):
        bceloss = torch.nn.BCELoss(reduction='none')
        cross_entropy_loss = bceloss(y_pred, y_true)
        p_t = ((y_true * y_pred) +
               ((1 - y_true) * (1 - y_pred)))
        modulating_factor = 1.0
        if self._gamma:
            modulating_factor = torch.pow(1.0 - p_t, self._gamma)
        alpha_weight_factor = 1.0
        if self._alpha is not None:
            alpha_weight_factor = (y_true * self._alpha +
                                   (1 - y_true) * (1 - self._alpha))
        focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor *
                                    cross_entropy_loss)
        return focal_cross_entropy_loss.mean()