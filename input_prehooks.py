import torch

# Generate forward pre hooks to record the input into features dict
# Features is a dictionary for module inputs
    # Key is module name
    # Value is module input
# Function does NOT actually modify features, only puts hooks that will eventually 
# modify features when input is later fed into the model
# specify device, have tensors on not cuda device so it doesn't hog vram
def put_input_hooks(model, features, feature_storage_device, verbose=False):

    # Function to make a hook function that inserts input into features dictionary
    def get_features(name):

        # pre_hook function that is input into nn.module.register_forward_pre_hook
        def pre_hook(model, input):
            if verbose:
                try:
                    print(f"for input {name}, shape is {input[0].shape}")
                except:
                    pass
            # move tensors of input to device, possibly store on different device with more memory
            
            if len(input) > 0:
                # concatenate with self
                if name in features:
                    features[name] = torch.cat((features[name], input[0].to(device=feature_storage_device)), dim=0)
                # make new entry if not existing
                else:
                    features[name] = input[0].to(device=feature_storage_device)

        
        # return the pre_hook function that will be fed into register_forward_pre_hook
        return pre_hook
    
    # call get_features, put in hooks at every module
    for n, m in model.named_modules():
        new_hook = get_features(n)
        # print(m)
        m.register_forward_pre_hook(new_hook)

