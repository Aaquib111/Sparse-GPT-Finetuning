# LOAD PRUNED MODEL

import torch
from torch.nn.utils import prune
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# function to get module name from parameter name
def get_module_name(param_name):
    if param_name[-5:] == ".bias":
        return param_name[:-5], "bias"
    elif param_name[-7:] == ".weight":
        return param_name[:-7], "weight"
    
    elif param_name[-10:] == ".bias_orig":
        return param_name[:-10], "bias"
    elif param_name[-12:] == ".weight_orig":
        return param_name[:-12], "weight"
    else:
        return None, None

# load model without masks
def load_unmasked_model(existing_model, state_dict_path):
    existing_model.load_state_dict(torch.load(state_dict_path))

# prune 0s to a mask, to make training easier (ostensibly)
class ThresholdPruning(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    # default threshold is 0, prunes weights that are already 0 (for training)
    def __init__(self, threshold=0):
        self.threshold = threshold

    def compute_mask(self, tensor, default_mask):
        return torch.abs(tensor) >= self.threshold

# apply pytorch mask in place of 0 weights to make backpropagation easier for training
default_opt_blacklist = ['model.decoder.embed_tokens', 'model.decoder.embed_positions']
def mask_from_pruned(model, module_blacklist=default_opt_blacklist):
    module_dict = {}
    for n, m in model.named_modules():
        module_dict[n] = m
    # print(module_dict.keys())
    
    parameter_list = []
    param_dict = {}
    for n, m in model.named_parameters():
        parameter_list.append(n)
        param_dict[n] = m
    # print(parameter_list)

    for n in parameter_list:
        module_name, param_type = get_module_name(n)

        # skip bias, embed, etc parameters
        if module_name in module_blacklist or module_name is None \
            or param_type is None or param_type!="weight":
            continue

        if len(param_dict[n].shape) < 2:
            continue

        ThresholdPruning.apply(module=module_dict[module_name], name=param_type)

# unmask model with 0s in place
def unmask_model(model, module_blacklist=default_opt_blacklist):
    module_dict = {}
    for n, m in model.named_modules():
        module_dict[n] = m
    # print(module_dict.keys())
    
    parameter_list = []
    param_dict = {}
    for n, m in model.named_parameters():
        parameter_list.append(n)
        param_dict[n] = m
    # print(parameter_list)

    for n in parameter_list:
        module_name, param_type = get_module_name(n)

        # skip bias, embed, etc parameters
        if module_name in module_blacklist or module_name is None \
            or param_type is None or param_type!="weight":
            continue

        if len(param_dict[n].shape) < 2:
            continue
            
        prune.remove(module=module_dict[module_name], name=param_type)

# load model with masks
def load_masked_model(existing_model, state_dict_path):

    # first load like normal
    load_unmasked_model(existing_model, state_dict_path)
    
    # then reapply the (previously removed) masks
    mask_from_pruned(model=existing_model)

    # prune.global_unstructured(
    #     existing_model.parameters(), pruning_method=ThresholdPruning, threshold=0
    # )


'''
# Iterate through all layers of preloaded model and apply the identity mask
def apply_identity_prune(model):
    module_lookup_dict = {}
    for module_name, module_iter in model.named_modules():
        module_lookup_dict[module_name] = module_iter

    # Using calibration data (inputs to each intermediate weight layer)
    # Iterate through named parameters, calculate inverse hessian and calculate mask

    # without this
    param_lookup_dict = {}
    param_names = []
    for name, param in model.named_parameters():
        param_names.append(name)
        param_lookup_dict[name] = param

    with torch.no_grad():
        for name in tqdm(param_names):
            param = param_lookup_dict[name]

            if 'embed' in name:
                continue
            
            # skip norms which have 1 dimension
            if len(param.shape) < 2:
                continue
            
            module_name, param_type = get_module_name(name)
            module = module_lookup_dict[module_name]
            prune.identity(module=module, name=param_type)

'''
