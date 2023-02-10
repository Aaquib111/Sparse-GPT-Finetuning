import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inverse_hessian(X, epsilon=0.5):
    """
    Calculate the inverse of a positive-definite matrix using the Cholesky decomposition.
    Args:
    - Hessian (torch.Tensor): positive-definite matrix to be inverted
    - epsilon (float): small constant to prevent Hessian from being singular
    Returns:
    - torch.Tensor: inverted matrix
    """
    X = X.float()
    X = torch.transpose(X, 0, 1)
    X = (2 * (X @ torch.transpose(X, 0, 1)))
    X = X + torch.eye(X.shape[0]) * epsilon
    
    # print(f"num 0s: {torch.sum(X.diag()==0)}")
    
    print(f"sum of diagonal {torch.sum(X.diag())}")
    # print(f"determinant: {torch.linalg.det(X)}")

    hessian = np.linalg.inv(X)
    # Decompose the matrix into a upper triangular matrix
    inverse_hessian = torch.transpose(torch.cholesky(hessian, upper=True), 0, 1)
    return inverse_hessian