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
    print(f"input shape: {X.shape}")

    if flattened:
        X_T = torch.transpose(X, 0, 1)
        identity = torch.eye(X.shape[0], dtype=torch.float64)
        # print(f"shape of x @ x_t: {torch.sum(X @ X_T, dim=0).shape}")
        H = 2 * (X @ X_T + (epsilon * identity))
    else:
        X_T = torch.transpose(X, 1, 2)
        identity = torch.eye(X.shape[1], dtype=torch.float64)
        # print(f"shape of x @ x_t: {torch.sum(X @ X_T, dim=0).shape}")
        H = 2 * (torch.sum(X @ X_T, dim=0) + (epsilon * identity))
    # print(torch.linalg.eig(H)[0])
    print(f"H SHAPE: {H.shape}")
    # print(f"num zeros in hessian: {torch.sum(H == 0)}")
    # print(f"Determinant is {torch.linalg.det(H)}")
    # print(f"Hessian Diagonal is {H.diag()}")
    H_inv = torch.inverse(H)
    
    # H_inv = torch.cholesky(H_inv).T
    H_inv = torch.lu(H_inv)[0].T
    
    return H_inv