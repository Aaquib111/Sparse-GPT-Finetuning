import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inverse_hessian(X, epsilon=0.01, flattened=False):
    """
    Calculate the inverse of a positive-definite matrix using the Cholesky decomposition.
    Args:
    - X (torch.Tensor): dxn tensor
    - epsilon (float): small constant to prevent Hessian from being singular
    Returns:
    - torch.Tensor: inverted matrix
    """
    X = X.double()
    #print(f"hessian input shape: {X.shape}")

    if flattened:
        X_T = torch.transpose(X, 0, 1)
        identity = torch.eye(X.shape[0], dtype=torch.float64)
        # print(f"shape of x @ x_t: {torch.sum(X @ X_T, dim=0).shape}")
        H = 2 * (X @ X_T) + (epsilon * identity)
    else:
        X_T = torch.transpose(X, 1, 2)
        identity = torch.eye(X.shape[1], dtype=torch.float64)
        # print(f"shape of x @ x_t: {torch.sum(X @ X_T, dim=0).shape}")
        H = 2 * (torch.sum(X @ X_T, dim=0)) + (epsilon * identity)
    
    triangle = torch.linalg.cholesky(H, upper=True)
    triangle_inv = torch.linalg.solve_triangular(identity, triangle, upper=True)[0]
    H_inv = triangle_inv @ triangle_inv.t()

    return H_inv