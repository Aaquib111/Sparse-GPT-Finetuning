# Generate forward pre hooks to record the input into features dict
# Features is a dictionary for module inputs
    # Key is module name
    # Value is module input
# Function does NOT actually modify features, only puts hooks that will eventually 
# modify features when input is later fed into the model
def put_input_hooks(model, features, verbose=False):

    # Function to make a hook function that inserts input into features dictionary
    def get_features(name):

        # pre_hook function that is input into nn.module.register_forward_pre_hook
        def pre_hook(model, input):
            if verbose:
                try:
                    print(f"for input {name}, shape is {input[0].shape}")
                except:
                    pass
            features[name] = input
        
        # return the pre_hook function that will be fed into register_forward_pre_hook
        return pre_hook
    
    # call get_features, put in hooks at every module
    for n, m in model.named_modules():
        new_hook = get_features(n)
        # print(m)
        m.register_forward_pre_hook(new_hook)

