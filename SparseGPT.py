import torch
from torch.nn.utils import prune

from transformers import AutoTokenizer, OPTForCausalLM, pipeline
from datasets import load_dataset

#from calculate_mask import calculate_mask
#from inverse_hessian import inverse_hessian
from input_prehooks import put_input_hooks


#DEVICE
device = 'cuda' if torch.cuda.is_available() else 'cpu'


#Load dataset
dataset = load_dataset('c4', 'en', streaming=True)
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
# Load model with pre-trained head
model = OPTForCausalLM.from_pretrained("facebook/opt-125m", output_attentions=True, output_hidden_states=True)
# Load genrator
generator = pipeline('text-generation', model="facebook/opt-125m")
# Create calibration data
calibration_data = []
for i, data in enumerate(iter(dataset['train'])):
    if i > 7:
        break
    tokenized = tokenizer.encode(data['text'], return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    calibration_data.append(tokenized)
#calibration_data = torch.transpose(torch.squeeze(torch.stack(calibration_data)),0,1).to(device=device)
calibration_data = torch.squeeze(torch.stack(calibration_data)).to(device=device)
calibration_data.double()


# First, put in forward hooks
features = {}
put_input_hooks(model=model, features=features)

# I drink for fun not to get rid of pain
# Run calibration data through model at first to calculate features dictionary with
# input tensors to each intermediate layer
model(calibration_data)

# function to get module name from parameter name
def get_module_name(param_name):
    if param_name[-5:] == ".bias":
        return param_name[:-5], "bias"
    elif param_name[-7:] == ".weight":
        return param_name[:-7], "weight"
    else:
        return None, None

def calculate_mask(
    W,
    H_inv,
    p,
    B,
    Bs,
    ):

    # Hi Aaquib

    # Get the number of rows and columns in W
    (d_row, d_col) = W.shape

    # Initialize the pruning mask M and block quantization errors E to all zeros

    M = torch.zeros(d_row, d_col, dtype=torch.bool)
    E = torch.zeros(d_row, B, dtype=torch.float64)

    # only need to calculate w_square and h_square once
    # Loop over blocks of columns of W (as specified by B)

    for i in range(0, d_col, B):

        # Loop over columns within a block

        for j in range(i, min(i + B - 1, d_col)):

            # If j is a multiple of Bs, prune a portion of the weights

            if j % Bs == 0:

                # Get the mask for the largest (1 - p)% of weights based on squared value and inverse hessian

                # ASTERISK: prune_values is matrix of w^2/H^(-1)_cc

                # Finding respective sections of hessian and weights matrix
                w_square_section = torch.square(W[:, j:j + Bs])
                h_square_section = torch.square(H_inv[j:j + Bs, j:j + Bs]).diag()  # 1 dimensional vector

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
            print(W[:, j])
            if torch.isnan(E[:, j-i]).sum() > 0:
                print(E[:,j-i])
                print(W[:5, j])
                print(H_inv[j,j])

            # Freeze the weights that are not pruned by multiplying by the pruning mask
            # Invert mask (~M equivalent to 1 - M < might be -(M + 1))

            E[:, j - i] = (~M[:, j]) * E[:, j - i]

            # Update the weights in this block based on the pruning error and inverse hessian information
            #print(torch.ger(E[:, j - i], H_inv[j, j:i + B]).shape)
            #print(torch.isnan(torch.ger(E[:, j - i], H_inv[j, j:i + B])).sum())
            W[:, j:(i + B)] = W[:, j:(i + B)] - torch.ger(E[:,(j - i)], H_inv[j, j:(i + B)])

            # Elite hacking

            # More elite hacking

            # Hacking x3

            # hacking hacking hacking

        # Update all remaining weights
    
        # print(f"this weight shape: {W[:, i + B:].shape}")
        # print(f"e shape: {E.shape}")
        # print(f"Hessian shape: {H_inv[i:i + B, i + B:].shape}")
        W[:, (i + B):] = W[:, (i + B):] - torch.matmul(E, H_inv[i:(i + B), (i + B):])
        if torch.isnan(W).sum() > 0:
            print(i, j)
            print(E)
            print(H_inv[i:i + B, i + B:])
            print(torch.isnan(E).sum())
            print(torch.isnan(H_inv[i:i + B, i + B:]).sum())
            print(torch.matmul(E, H_inv[i:i + B, i + B:]).shape)
            print(torch.isnan(torch.matmul(E, H_inv[i:i + B, i + B:])).sum())
            print(torch.isnan(W).sum())
    
    print(torch.isnan(W).sum())

    # return mask

    return M


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


# Re-load model with pre-trained head
model = OPTForCausalLM.from_pretrained("facebook/opt-125m", output_attentions=True, output_hidden_states=True)

# make a dictionary to access module by name
module_lookup_dict = {}
for module_name, module_iter in model.named_modules():
    module_lookup_dict[module_name] = module_iter
EPSILON = 1
SPARSENESS = .2
B = 32
Bs = 16

layer_blacklist = ['model.decoder.embed_tokens.weight', 'model.decoder.embed_tokens.bias',
'model.decoder.embed_positions.weight', 'model.decoder.final_layer_norm.weight',
'model.decoder.final_layer_norm.bias']

# Using calibration data (inputs to each intermediate weight layer)
# Iterate through named parameters, calculate inverse hessian and calculate mask

# without this
param_lookup_dict = {}
param_names = []
for name, param in model.named_parameters():
    param_names.append(name)
    param_lookup_dict[name] = param

with torch.no_grad():
    for name in param_names:
        param = param_lookup_dict[name]

        # skip the embed layer
        if name in layer_blacklist:
            continue
        
        # skip norms which have 1 dimension
        if len(param.shape) < 2:
            continue

        module_name, param_type = get_module_name(name)

        # apply to weight and bias layers
        if param_type == "weight" or param_type == "bias":
            # input to parameter
            layer_input = features[module_name][0]
            print(name)
            print(f"layer input shape: {layer_input.shape}")
            # print(f"weight shape: {param.shape}")

            # calculate inverse hessian
            # check if input is flattened e.g. from 8,512,768 to 4096,768
            if len(layer_input.shape) == 2:
                inv_hess = inverse_hessian(torch.transpose(layer_input, 0, 1), epsilon=EPSILON, 
                flattened=True)

            else:
                inv_hess = inverse_hessian(torch.transpose(layer_input, 1, 2), epsilon=EPSILON,
                flattened=False)

            # inv_hess = inverse_hessian(layer_input, epsilon=EPSILON)
            # print(f"hessian shape: {inv_hess.shape}")

            # calculate mask
            mask = calculate_mask(W=param, H_inv=inv_hess, p=SPARSENESS, B=B, Bs=Bs)
            
            # get module from lookup dictionary by module name
            module = module_lookup_dict[module_name]
            # apply mask
            prune.custom_from_mask(module=module, name=param_type, mask=mask)
        # break

test_param = model.get_decoder().layers[0].self_attn.k_proj.weight_orig
# for name, test_param in model.named_parameters():
#     param = param_lookup_dict[name]

#     # skip the embed layer
#     if name in layer_blacklist:
#         continue
    
#     # skip norms which have 1 dimension
#     if len(param.shape) < 2:
#         continue

#     print(torch.isnan(test_param).sum())
# print(torch.max(test_param))
print(torch.sum(test_param == 0.0))
print(test_param)
# print(test_param.shape)