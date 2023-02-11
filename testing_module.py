# manage imports
import torch
from transformers import AutoTokenizer, OPTForCausalLM, pipeline
from datasets import load_dataset

# model, data settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = load_dataset('c4', 'en', streaming=True)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

#handling input data
model = OPTForCausalLM.from_pretrained("facebook/opt-125m", output_attentions=True, output_hidden_states=True)
generator = pipeline('text-generation', model="facebook/opt-125m")
input_data = []
for i, data in enumerate(iter(dataset['train'])):
    if i > 7:
        break
    tokenized = tokenizer.encode(data['text'], return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    input_data.append(tokenized)

def calculate_perp(model, input_data, device):
    input_data = torch.squeeze(torch.stack(input_data)).to(device=device)
    input_data.double()
    outputs = model(input_data)[0] 
    log_probs = outputs[0, -1, :].log_softmax(-1)
    neg_log_likelihood = -log_probs.mean()
    perplexity = torch.exp(neg_log_likelihood)      
    return perplexity.item()

print(calculate_perp(model, input_data, device))



