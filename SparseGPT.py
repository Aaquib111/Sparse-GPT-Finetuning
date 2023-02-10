import torch
from torch.nn.utils import prune

from transformers import AutoTokenizer, OPTForCausalLM, pipeline
from datasets import load_dataset

from calculate_mask import calculate_mask
from inverse_hessian import inverse_hessian
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
    if i > 1:
        break
    tokenized = tokenizer.encode(data['text'], return_tensors="pt", padding="max_length", truncation=True, max_length=2048)
    calibration_data.append(tokenized)
calibration_data = torch.squeeze(torch.stack(calibration_data)).to(device=device)



# First, put in forward hooks
features = {}
put_input_hooks(model=model, features=features)

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


# make a dictionary to access module by name
model_lookup_dict = {}
for module_name, module_iter in model.named_modules():
    model_lookup_dict[module_name] = module_iter

EPSILON = 1e-8
SPARSENESS = .2
B = 32
Bs = 16

# Using calibration data (inputs to each intermediate weight layer)
# Iterate through named parameters, calculate inverse hessian and calculate mask
for name, param in model.named_parameters():
    module_name, param_type = get_module_name(name)

    # apply to weight and bias layers
    if param_type == "weight" or param_type == "bias":
        # input to parameter
        layer_input = features[module_name]
        # calculate inverse hessian
        inv_hess = inverse_hessian(layer_input, epsilon=EPSILON)

        # calculate mask
        mask = calculate_mask(W=param, H_inv=inv_hess, p=SPARSENESS, B=B, Bs=Bs)
        
        # get module from lookup dictionary by module name
        module = model_lookup_dict[module_name]
        # apply mask
        prune.custom_from_mask(module=module, name=param_type, mask=mask)
    break
