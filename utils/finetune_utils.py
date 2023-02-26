# manage imports
import torch
from transformers import DataCollatorForLanguageModeling,AutoTokenizer, OPTForCausalLM, pipeline
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

from utils.save_utils import load_unmasked_model, load_masked_model, unmask_model, mask_from_pruned
from utils.prehook_utils import remove_all_hooks
import gc
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 

def finetune_model(model_name, tokenizer, SPARSITY, device=DEVICE, EPOCH_COUNT=10):
    #encode tokens
    def encode_tok(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length')

    #stream c4, training split
    training_data = load_dataset('c4', 'en', split='train', streaming=True)
    #IMPORTANT: process data while streaming -> remove unnecessary columns in batches
    training_data = training_data.map(encode_tok, batched=True, remove_columns=["text", "timestamp", "url"])

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
    remove_all_hooks(loaded_model)
    loaded_model.eval()
    _ = loaded_model(torch.randint(high=20, size=(1,10)))

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
    
def finetune_model_inplace(model, tokenizer, SPARSITY, device=DEVICE, EPOCH_COUNT=10):
    #encode tokens
    def encode_tok(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length')

    #stream c4, training split
    training_data = load_dataset('c4', 'en', split='train', streaming=True)
    #IMPORTANT: process data while streaming -> remove unnecessary columns in batches
    training_data = training_data.map(encode_tok, batched=True, remove_columns=["text", "timestamp", "url"])

    #set data to tensor mode
    training_data = training_data.with_format("torch")

    #dataloader from dataloader (mlm=False when training without mask)
    reformatted_data = DataLoader(training_data, collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False))
    
    #training loop
    model.train().to(device)
    t_optim = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
    for epoch in tqdm(range(EPOCH_COUNT)):
        training_data.set_epoch(epoch)
        for i, batch in enumerate(reformatted_data):
            if i == 5:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            print(f"max memory during batch: {torch.cuda.memory_allocated()}")
            loss = outputs[0]
            loss.backward()
            t_optim.step()
            t_optim.zero_grad()
        
        print(f"memory allocated after epoch: {torch.cuda.memory_allocated()}")
    #unmask_model(model)
