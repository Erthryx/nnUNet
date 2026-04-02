import torch
import torch.nn as nn
import torch.nn.functional as F

class InstanceFbetaLoss(nn.Module):
    """
    Differentiable soft F-beta loss for instance-level segmentation.
    Approximates instance F0.5 score using predicted probabilities (soft masks)
    and ground truth masks, fully differentiable.
    """
    def __init__(self, beta=0.5, eps=1e-6):
        """
        :param beta: F-beta parameter; beta < 1 emphasizes precision, beta > 1 emphasizes recall
        :param eps: small value to avoid division by zero
        """
        super().__init__()
        self.beta2 = beta**2
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        """
        :param logits: network output, shape (B, C, H, W), raw logits
        :param target: ground truth, shape (B, C, H, W), one-hot encoded
        :return: scalar loss (to minimize)
        """
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=1)

        # Flatten batch & spatial dims for calculation
        probs_flat = probs.view(probs.size(0), probs.size(1), -1)
        target_flat = target.view(target.size(0), target.size(1), -1)

        # Compute soft true positives, false positives, false negatives
        tp = (probs_flat * target_flat).sum(dim=2)
        fp = (probs_flat * (1 - target_flat)).sum(dim=2)
        fn = ((1 - probs_flat) * target_flat).sum(dim=2)

        # Soft F-beta
        fbeta = (1 + self.beta2) * tp / ((1 + self.beta2) * tp + self.beta2 * fn + fp + self.eps)

        # Convert to loss
        loss = 1 - fbeta.mean()  # mean over batch and classes
        return loss