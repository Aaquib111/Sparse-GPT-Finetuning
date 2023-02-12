# LOAD PRUNED MODEL

import torch
from torch.nn.utils import prune
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'



def load_into_model(existing_model, state_dict_path):
    apply_identity_prune(model=existing_model)

    existing_model.load_state_dict(torch.load(state_dict_path))


# function to get module name from parameter name
def get_module_name(param_name):
    if param_name[-5:] == ".bias":
        return param_name[:-5], "bias"
    elif param_name[-7:] == ".weight":
        return param_name[:-7], "weight"
    else:
        return None, None
        
# Iterate through all layers of preloaded model and apply the identity mask
def apply_identity_prune(model):
    module_lookup_dict = {}
    for module_name, module_iter in model.named_modules():
        module_lookup_dict[module_name] = module_iter

    layer_blacklist = ['loaded_model.decoder.embed_tokens.weight', 'loaded_model.decoder.embed_tokens.bias',
    'loaded_model.decoder.embed_positions.weight']

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

            # skip the embed layer
            # if name in layer_blacklist:
            #     continue

            if 'embed' in name:
                continue
            
            # skip norms which have 1 dimension
            if len(param.shape) < 2:
                continue
            
            module_name, param_type = get_module_name(name)
            module = module_lookup_dict[module_name]
            prune.identity(module=module, name=param_type)