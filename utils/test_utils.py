# manage imports
import torch
from transformers import DataCollatorForLanguageModeling,AutoTokenizer, OPTForCausalLM, pipeline
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from save_pruned_model import load_unmasked_model, load_masked_model, unmask_model, mask_from_pruned
import gc

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

def test_model(model_name, encodings, token_length, seq_len, stride, wandb, SPARSITY, is_finetuned=False, device=device):
    loaded_model = OPTForCausalLM.from_pretrained(f'facebook/{model_name}',
                                                  output_attentions=True,
                                                  output_hidden_states=True).to(device=device) # type: ignore
    loaded_model = torch.nn.DataParallel(loaded_model, device_ids=[0,1,2,3])
    
    if is_finetuned:
        load_unmasked_model(loaded_model, 
                            f'pruned_models/{model_name}-finetuned-{SPARSITY}.pt')
    else:
        if SPARSITY != 1:
            load_unmasked_model(loaded_model, 
                            f'pruned_models/{model_name}-{SPARSITY}.pt')
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


def finetune_model(model_name, tokenizer, SPARSITY, device=device, EPOCH_COUNT=10):
    #encode tokens
    def encode_tok(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length')

    #stream c4, training split
    training_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', streaming=True)
    #IMPORTANT: process data while streaming -> remove unnecessary columns in batches
    training_data = training_data.map(encode_tok, batched=True, remove_columns=["text"])

    #set data to tensor mode
    training_data = training_data.with_format("torch")

    #dataloader from dataloader (mlm=False when training without mask)
    reformatted_data = DataLoader(training_data, collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False))
    
    loaded_model = OPTForCausalLM.from_pretrained(f'facebook/{model_name}',
                                                  output_attentions=True,
                                                  output_hidden_states=True).to(device=device) # type: ignore
    loaded_model = torch.nn.DataParallel(loaded_model, device_ids=[0,1,2,3])# activate masks
    
    if SPARSITY != 1:
        load_masked_model(loaded_model, f'pruned_models/{model_name}-{SPARSITY}.pt')
    loaded_model.eval()
    _ = loaded_model(torch.randint(high=20, size=(1,10)))
    mask_from_pruned(model=loaded_model)
    #training loop
    loaded_model.train().to(device)
    t_optim = torch.optim.AdamW(params=loaded_model.parameters(), lr=1e-5)
    for epoch in tqdm(range(EPOCH_COUNT)):
        training_data.set_epoch(epoch)
        for i, batch in enumerate(reformatted_data):
            if i == 5:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = loaded_model(**batch)
            loss = outputs[0]
            loss.backward()
            t_optim.step()
            t_optim.zero_grad()
    unmask_model(loaded_model)
    torch.save(loaded_model.state_dict(), f'pruned_models/{model_name}-{SPARSITY}-finetuned.pt')

