import torch.nn.utils.prune as prune

def get_module_name(param_name):
    if param_name[-5:] == ".bias":
        return param_name[:-5], "bias"
    elif param_name[-7:] == ".weight":
        return param_name[:-7], "weight"
    else:
        return None, None

default_opt_blacklist = ['model.decoder.embed_tokens', 'model.decoder.embed_positions']

# mask the lowest magnitude weights of the whole model
def mask_lowest(model, amount=.2, module_blacklist=default_opt_blacklist, prune_remove=True):
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


