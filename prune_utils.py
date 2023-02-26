from input_prehooks import get_feature_storage_name
import gc
import torch
from tqdm import tqdm
from torch.nn.utils import prune
from utils.hessian_utils import calc_inverse_hessian
from utils.mask_utils import calculate_mask

opt_blacklist = ['module.model.decoder.embed_tokens', 'module.model.decoder.embed_positions']

#DEVICE
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Using calibration data (inputs to each intermediate weight layer)
# Iterate through named parameters, calculate inverse hessian and calculate mask

def get_module_name(param_name):
    if param_name[-5:] == ".bias":
        return param_name[:-5], "bias"
    elif param_name[-7:] == ".weight":
        return param_name[:-7], "weight"
    else:
        return None, None


def sparsegpt_prune(model, model_name, feature_hessians, EPSILON, 
                    SPARSENESS, B, Bs, module_blacklist=opt_blacklist, save_model=True):
    module_dict = {}
    for n, m in model.named_modules():
        module_dict[n] = m
    # print(module_dict.keys())
    
    param_names = []
    param_dict = {}
    for n, m in model.named_parameters():
        param_names.append(n)
        param_dict[n] = m
    # print(parameter_list)

    model.eval()
    with torch.no_grad():
        # for name in tqdm(param_names):
        for param_name in tqdm(param_names, total=len(param_names)):
            module_name, param_type = get_module_name(param_name)

            # skip bias, embed, etc parameters
            if module_name in module_blacklist or module_name is None \
                or param_type is None or param_type!="weight":
                continue

            if len(param_dict[param_name].shape) < 2:
                continue

            param = param_dict[param_name]

            #print(f"Doing layer {name}")
            # get layer input from features, key is get_feature_storage_name(module_name)
            # get_feature_storage_name(module_name) stores k_proj, v_proj, q_proj together
            # since they are the same input
            
            layer_hessian = feature_hessians[get_feature_storage_name(module_name)].to(device=device)

            # calculate inverse hessian
            # check if input is flattened e.g. from 8,512,768 to 4096,768
            try:
                inv_hess = calc_inverse_hessian(layer_hessian)
            except:
                print(f'Cant prune {param_name}')
                continue

            # calculate mask
            mask = calculate_mask(W=param, H_inv=inv_hess, p=SPARSENESS, B=B, Bs=Bs)
            
            # get module from lookup dictionary by module name
            module = module_dict[module_name]
            # apply mask
            prune.custom_from_mask(module=module, name=param_type, mask=mask)
            prune.remove(module=module, name=param_type)
            gc.collect()
            torch.cuda.empty_cache()  

    pruned_model_name = f'{model_name}-{SPARSENESS}'
    if save_model:
        torch.save(model.state_dict(), f'pruned_models/{pruned_model_name}.pt')
    
# mask the lowest magnitude weights of the whole model
def mask_lowest(model, amount=.2, module_blacklist=opt_blacklist, prune_remove=True):
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
        if module_name in default_opt_blacklist or module_name is None \
            or param_type is None or param_type!="weight":
            continue

        if len(param_dict[n].shape) < 2:
            continue

        # perform the masking
        module = module_dict[module_name]
        # param_type should always be 'weight'
        prune.l1_unstructured(module=module, name=param_type, amount=amount)

        # remove mask, apply mask and make some weights 0
        if prune_remove:
            prune.remove(module=module, name=param_type)
