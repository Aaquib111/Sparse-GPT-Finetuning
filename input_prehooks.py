import torch

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
'self_attn.v_proj',
'self_attn.q_proj',
'self_attn.out_proj',
'fc1',
'fc2']


def put_input_hooks(model, features, feature_storage_device, verbose=False, whitelist=opt_whitelist):

    # Function to make a hook function that inserts input into features dictionary
    def get_features(name):
        print(name)

        # pre_hook function that is input into nn.module.register_forward_pre_hook
        def pre_hook(model, input):
            if verbose:
                try:
                    print(f"for input {name}, shape is {input[0].shape}")
                except:
                    pass

            # move tensors of input to device, possibly store on different device with more memory
            # check whitelist
            if len(input) > 0 and check_whitelist(name, whitelist):
                # get name as key to store in features (since k_proj, q_proj, v_proj have same input)
                storage_name = get_feature_storage_name(name)
                # concatenate with self
                if storage_name in features:
                    features[storage_name] = torch.cat((features[storage_name], input[0].to(device=feature_storage_device)), dim=0)
                # make new entry if not existing
                else:
                    features[storage_name] = input[0].to(device=feature_storage_device)

        
        # return the pre_hook function that will be fed into register_forward_pre_hook
        return pre_hook
    
    # call get_features, put in hooks at every module
    for n, m in model.named_modules():
        new_hook = get_features(n)
        # print(m)
        m.register_forward_pre_hook(new_hook)

