import torch
from inverse_hessian import calc_hessian

# Generate forward pre hooks to record the input into features dict
# Features is a dictionary for module inputs
    # Key is module name
    # Value is module input
# Function does NOT actually modify features, only puts hooks that will eventually 
# modify features when input is later fed into the model
# specify device, have tensors on not cuda device so it doesn't hog vram
# only store key, query, value proj inputs, out_proj inputs, 


def get_feature_storage_name(param_name):
    if 'k_proj' in param_name:
        param_name = param_name.replace("k_proj", "in_proj")

    elif 'v_proj' in param_name:
        param_name = param_name.replace("v_proj", "in_proj")

    elif 'q_proj' in param_name:
        param_name = param_name.replace("q_proj", "in_proj")
    return param_name


# Function to check if param should be added to features dictionary
def check_whitelist(param_name, whitelist):
    for name_end in whitelist:
        if name_end in param_name:
            return True

    return False


opt_whitelist = ['self_attn.k_proj',
'decoder.project_out',
'decoder.project_in',
# 'self_attn.v_proj',
# 'self_attn.q_proj',
'self_attn.out_proj',
'fc1',
'fc2']


def put_input_hooks(model, features, feature_storage_device, verbose=False, whitelist=opt_whitelist):#, storage_dir, offload_freq=16

    # Function to make a hook function that inserts input hessian into features dictionary
    def get_features(name):

        # pre_hook function that is input into nn.module.register_forward_pre_hook
        def pre_hook(model, input):
            if verbose:
                try:
                    print(f"for input {name}, shape is {input[0].shape}")
                except:
                    pass

            # move tensors of input hessian to device, possibly store on different device with more memory
            # check whitelist
            if len(input) > 0 and check_whitelist(name, whitelist):
                # get name as key to store in features (since k_proj, q_proj, v_proj have same input)
                storage_name = get_feature_storage_name(name)
                #print(storage_name)
                # check if flattened
                if len(input[0].shape) == 2:
                    input_hessian = calc_hessian(torch.transpose(input[0], 0, 1), flattened=True).to(device=feature_storage_device)
                # not flattened
                else:
                    input_hessian = calc_hessian(torch.transpose(input[0], 1, 2), flattened=False).to(device=feature_storage_device)

                if storage_name in features:
                    features[storage_name] += input_hessian
                # make new entry if not existing
                else:
                    features[storage_name] = input_hessian
                    
                torch.cuda.empty_cache()
                '''if features[storage_name].shape[0] % offload_freq == 0:
                    if os.path.exists(f'{storage_dir}/{name}'):
                        existing_tensor = torch.load(f'{storage_dir}/{name}')
                        new_tensor = torch.cat((existing_tensor, features[name]), dim=0)
                        torch.save(new_tensor,f'{storage_dir}/{name}')
                    # make new entry if not existing
                    else:
                        torch.save(features[storage_name],f'{storage_dir}/{name}')
                    del features[storage_name]'''

        
        # return the pre_hook function that will be fed into register_forward_pre_hook
        return pre_hook
    
    # call get_features, put in hooks at every module
    for n, m in model.named_modules():
        new_hook = get_features(n)
        # print(m)
        m.register_forward_pre_hook(new_hook)

