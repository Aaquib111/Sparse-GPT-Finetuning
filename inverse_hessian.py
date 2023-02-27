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
def calc_inverse_hessian(H, epsilon=0.01):
    """
    Calculate the inverse of a positive-definite matrix using the Cholesky decomposition.
    Args:
    - X (torch.Tensor): dxn tensor
    - epsilon (float): small constant to prevent Hessian from being singular
    Returns:
    - torch.Tensor: inverted matrix
    """

    return torch.cholesky_inverse(torch.linalg.cholesky(H, upper=True), upper=True)


# testing adding hessians
if __name__ == '__main__':

    test_tensor = torch.rand(2, 3, 5)
    flattened_tensor = torch.flatten(test_tensor, start_dim=0, end_dim=1)
    # print(inverse_hessian(torch.transpose(test_tensor, 1, 2), flattened=False))
    # print(inverse_hessian(torch.transpose(flattened_tensor, 0, 1), flattened=True))

    test_tensor_2 = torch.rand(2, 3, 5)
    flattened_tensor_2 = torch.flatten(test_tensor_2, start_dim=0, end_dim=1)

    # comb_tensor = torch.cat((test_tensor, test_tensor_2,))
    comb_tensor = torch.cat((flattened_tensor, flattened_tensor_2,))
    print(comb_tensor.shape)
    # flattened_tensor_3 = torch.flatten(comb_tensor, start_dim=0, end_dim=1)

    # print(inverse_hessian(torch.transpose(comb_tensor, 1, 2), flattened=False))
    # print(hessian(torch.transpose(flattened_tensor, 0, 1), flattened=True))
    # print(hessian(torch.transpose(flattened_tensor_2, 0, 1), flattened=True))
    # print(hessian(torch.transpose(comb_tensor, 0, 1), flattened=True))
    # print(hessian(torch.transpose(comb_tensor, 0, 1), flattened=True) - 
    #     (hessian(torch.transpose(flattened_tensor, 0, 1), flattened=True) + hessian(torch.transpose(flattened_tensor_2, 0, 1), flattened=True)))
