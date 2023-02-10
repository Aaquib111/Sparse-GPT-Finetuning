import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inverse_hessian(X, epsilon=0.01):
    """
    Calculate the inverse of a positive-definite matrix using the Cholesky decomposition.
    Args:
    - X (torch.Tensor): dxn tensor
    - epsilon (float): small constant to prevent Hessian from being singular
    Returns:
    - torch.Tensor: inverted matrix
    """
    X = X.float()
    X_T = torch.transpose(X, 0, 1)
    identity = torch.eye(X.shape[0], dtype=torch.float32)
    H_inv = torch.inverse(2 * (X @ X_T + epsilon * identity))
    #H_inv = torch.cholesky(H_inv).T
    H_inv = torch.lu(H_inv)[0].T
    
    return H_inv