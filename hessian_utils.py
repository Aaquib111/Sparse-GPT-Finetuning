import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# calculate hessian only WITHOUT EPSILON (epsilon changes addition formula, can use
# epsilon later on in inverse_hessian)
def calc_hessian(X, flattened=False, epsilon=0.01):
    X = X.double()

    if flattened:
        X_T = torch.transpose(X, 0, 1)
        identity = torch.eye(X.shape[0], dtype=torch.float32, device=device)
        # print(f"shape of x @ x_t: {torch.sum(X @ X_T, dim=0).shape}")
        H = 2 * (X @ X_T)
    else:
        X_T = torch.transpose(X, 1, 2)
        identity = torch.eye(X.shape[1], dtype=torch.float32, device=device)
        # print(f"shape of x @ x_t: {torch.sum(X @ X_T, dim=0).shape}")
        H = 2 * (torch.sum(X @ X_T, dim=0))
    H += (epsilon * identity)

    return H

# calculate inverse hessian from hessian, with some changes
def calc_inverse_hessian(H, epsilon=0.0001):
    """
    Calculate the inverse of a positive-definite matrix using the Cholesky decomposition.
    Args:
    - X (torch.Tensor): dxn tensor
    - epsilon (float): small constant to prevent Hessian from being singular
    Returns:
    - torch.Tensor: inverted matrix
    """
    try:
        return torch.cholesky_inverse(torch.linalg.cholesky(H, upper=True), upper=True)
    except:
        return torch.inverse(H)