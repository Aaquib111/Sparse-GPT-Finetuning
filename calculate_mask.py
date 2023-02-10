import torch as torch

'''
W is weights matrix for one layer
H_inv is inverse hessian for one layer
p is proportion of weights we're pruning (top p%)
B is lazy block size (low B helps to reduce memory use)
Bs is inverse of how often to make masks (e.g. when Bs is 4, make new masks with specified sparseness for every 4 columns)
'''

def calculate_mask(
    W_orig,
    H_inv,
    p,
    B,
    Bs,
    ):

    # Get the number of rows and columns in W

    (d_row, d_col) = W.shape

    # Initialize the pruning mask M and block quantization errors E to all zeros

    M = torch.zeros(d_row, d_col, dtype=torch.bool)
    E = torch.zeros(d_row, B)

    # only need to calculate w_square and h_square once
    # Loop over blocks of columns of W (as specified by B)

    for i in range(0, d_col, B):

        # Loop over columns within a block

        for j in range(i, min(i + B, d_col)):

            # If j is a multiple of Bs, prune a portion of the weights

            if j % Bs == 0:

                # Get the mask for the largest (1 - p)% of weights based on squared value and inverse hessian

                # ASTERISK: prune_values is matrix of w^2/H^(-1)_cc

                # Finding respective sections of hessian and weights matrix
                w_square_section = torch.square(W[:, j:j + Bs])
                h_square_section = torch.square(H_inv[j:j + Bs, j:j
                        + Bs]).diag()  # 1 dimensional vector

                # getting the prune values matrix from W and H^-1 sections
                prune_values = w_square_section \
                    / h_square_section.unsqueeze(0)

                #calulating cutoff for the weights
                cutoff_value = torch.kthvalue(prune_values, int((1 - p)
                        * d_row), dim=0)[0]

                #getting the final mask
                mask = prune_values > cutoff_value

                #masking
                M[:, j:j + Bs] = mask

            # Calculate the pruning error for this column

            E[:, j - i] = W[:, j] / H_inv[j, j]

            # Freeze the weights that are not pruned by multiplying by the pruning mask
            # Invert mask (~M equivalent to 1 - M < might be -(M + 1))

            E[:, j - i] = ~M[:, j] * E[:, j - i]

            # Update the weights in this block based on the pruning error and inverse hessian information

            W[:, j:i + B] -= torch.ger(E[:, j - i], H_inv[j, j:i + B])

        # Update all remaining weights

        W[:, i + B:] -= torch.matmul(E, H_inv[i:i + B, i + B:])

    # return mask

    return M