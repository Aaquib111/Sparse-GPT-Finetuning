# manage imports
import torch
from transformers import AutoTokenizer, OPTForCausalLM, pipeline
from datasets import load_dataset

# don't necessarily want this testing to be on cuda, maybe memory could be exceeded
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# method to calculate the perplexity of a model (loss function), meant for testing functionality of sparse models
def calculate_perp(model, input_data):
    return torch.exp(model(input_ids=input_data, labels=input_data).loss) 

if __name__ == '__main__':
    # model, data settings
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
    # print(calculate_perp(model, input_data, device))



