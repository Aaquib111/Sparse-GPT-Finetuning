# Playing around with sparse transformers

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

# Load model with pre-trained head
model = AutoModel.from_pretrained("facebook/opt-125m", output_attentions=True, output_hidden_states=True)

generator = pipeline('text-generation', model="facebook/opt-125m")

for name, param in model.named_parameters():
    print(name)
    print(param.shape)
