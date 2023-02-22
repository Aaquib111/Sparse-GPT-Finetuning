import torch.nn.utils.prune 
def get_module_name(param_name):
    if param_name[-5:] == ".bias":
        return param_name[:-5], "bias"
    elif param_name[-7:] == ".weight":
        return param_name[:-7], "weight"
    else:
        return None, None

# mask the lowest magnitude weights of the whole model
def mask_lowest(model, amount=.2):
    module_dict = {}
    for n, m in model.named_modules():
        module_dict[n] = m
    # print(module_dict.keys())
    
    parameter_list = []
    for n, m in model.named_parameters():
        parameter_list.append(n)
    # print(parameter_list)

    for n in parameter_list:
        module_name, param_type = get_module_name(n)
        if module_name is None or param_type is None or param_type=="bias":
            continue
        # perform the masking
        module = module_dict[module_name]
        torch.nn.utils.prune.l1_unstructured(module=module, name=param_type, amount=.2)