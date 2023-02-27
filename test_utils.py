# manage imports
import torch
from transformers import DataCollatorForLanguageModeling,AutoTokenizer, OPTForCausalLM, pipeline
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.save_utils import load_unmasked_model, load_masked_model, unmask_model, mask_from_pruned
import gc

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

def test_model(model_name, encodings, token_length, seq_len, stride, wandb, SPARSITY, model_type='finetuned', device=device):
    loaded_model = OPTForCausalLM.from_pretrained(f'facebook/{model_name}',
                                                  output_attentions=True,
                                                  output_hidden_states=True).to(device=device) # type: ignore
    loaded_model = torch.nn.DataParallel(loaded_model, device_ids=[0,1,2,3])
    
    if model_type == 'finetuned':
        load_unmasked_model(loaded_model, 
                            f'pruned_models/{model_name}-finetuned-{SPARSITY}.pt')
    elif model_type == 'pruned':
        if SPARSITY != 1:
            load_unmasked_model(loaded_model, 
                            f'pruned_models/{model_name}-{SPARSITY}.pt')
    elif model_type == 'iterative':
        if SPARSITY != 1:
            load_unmasked_model(loaded_model, 
                            f'pruned_models/{model_name}-finetuned-{SPARSITY}-iterative.pt')
    loaded_model.eval()
    _ = loaded_model(torch.randint(high=20, size=(1,10)))
    
    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + token_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device=device)
        target_ids = input_ids.clone()
        target_ids[:,:-trg_len] = -100
        
        with torch.no_grad():
            outputs = loaded_model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
            
        nlls.append(neg_log_likelihood)
        
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
            
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    wandb.log({"perplexity": ppl, 'density': SPARSITY})
    
    del loaded_model
    gc.collect()
    torch.cuda.empty_cache()
