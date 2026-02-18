"""
Utility functions for DLG attack on SCAFFOLD
"""
import torch
import torch.nn.functional as F


def label_to_onehot(target: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    """
    Convert label indices to one-hot encoded vectors.
    
    Args:
        target: Tensor of shape (batch_size,) containing label indices
        num_classes: Number of classes (default: 10 for CIFAR-10)
    
    Returns:
        One-hot encoded tensor of shape (batch_size, num_classes)
    
    Example:
        >>> target = torch.tensor([2, 5])
        >>> onehot = label_to_onehot(target, num_classes=10)
        >>> print(onehot.shape)
        torch.Size([2, 10])
    """
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


def cross_entropy_for_onehot(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy loss for one-hot encoded targets.
    
    This is useful when the target is already in one-hot format,
    which is the case in DLG attacks where we optimize dummy labels.
    
    Args:
        pred: Model predictions of shape (batch_size, num_classes)
        target: One-hot encoded targets of shape (batch_size, num_classes)
    
    Returns:
        Scalar tensor representing the cross-entropy loss
    
    Formula:
        CE = -sum(target * log_softmax(pred))
    
    Example:
        >>> pred = torch.randn(2, 10)
        >>> target = torch.zeros(2, 10)
        >>> target[0, 3] = 1  # Class 3 for first sample
        >>> target[1, 7] = 1  # Class 7 for second sample
        >>> loss = cross_entropy_for_onehot(pred, target)
    """
    return torch.mean(torch.sum(-target * F.log_softmax(pred, dim=-1), 1))


def compute_gradient_difference(grad1: list, grad2: list) -> torch.Tensor:
    """
    Compute the squared L2 distance between two gradient lists.
    
    Args:
        grad1: List of gradient tensors
        grad2: List of gradient tensors
    
    Returns:
        Scalar tensor representing ||grad1 - grad2||^2
    """
    grad_diff = 0
    for g1, g2 in zip(grad1, grad2):
        grad_diff += ((g1 - g2) ** 2).sum()
    return grad_diff


def total_variation(x: torch.Tensor) -> torch.Tensor:
    """
    Compute total variation regularization.
    
    This can be used to encourage smoothness in reconstructed images.
    TV(x) = sum(|x[i,j] - x[i+1,j]|) + sum(|x[i,j] - x[i,j+1]|)
    
    Args:
        x: Image tensor of shape (batch, channels, height, width)
    
    Returns:
        Scalar tensor representing the total variation
    """
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    
    # Horizontal variation
    h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :h_x-1, :]).sum()
    
    # Vertical variation
    w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x-1]).sum()
    
    return (h_tv + w_tv) / batch_size
