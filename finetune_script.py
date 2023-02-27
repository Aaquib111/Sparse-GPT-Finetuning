#imports
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, OPTForCausalLM, Trainer, TrainingArguments
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
from utils.save_utils import load_masked_model, load_masked_model_single

def training_step(model, training_data, optimizer):
    model.train()
    t_optim = torch.optim.AdamW(params=loaded_model.parameters(), lr=1e-5)
    
    training_data.set_epoch(epoch)
    for i, batch in enumerate(training_data):
        #print(batch)
        if i == 5:
            break
        batch = {k: torch.unsqueeze(torch.tensor(v), 0) for k, v in batch.items()}
        #print(batch)
        #print(batch['input_ids'].size())
        #print(batch['attention_mask '].size())
        outputs = model(**batch, labels=batch['input_ids'])
        loss = outputs.loss
        print(loss)
        accelerate.backward(loss)
        t_optim.step()
        t_optim.zero_grad()
        
def encode_tok(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length')

accelerate = Accelerator()
model_name='opt-1.3b'
EPOCH_COUNT=10
SPARSITY=0.2
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained(f'facebook/{model_name}', padding_side='left')

#stream c4, training split
training_data = load_dataset('c4', 'en', split='train', streaming=True)
#IMPORTANT: process data while streaming -> remove unnecessary columns in batches
training_data = training_data.map(encode_tok, 
                                  batched=True, 
                                  batch_size=2,
                                  remove_columns=["text", "timestamp", "url"])
#set data to tensor mode
training_data = training_data.with_format("torch")

#dataloader from dataloader (mlm=False when training without mask)
dataloader = DataLoader(training_data, 
                        collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False),
                        batch_size=4)

loaded_model = OPTForCausalLM.from_pretrained(f'facebook/{model_name}',
                                                  output_attentions=True,
                                                  output_hidden_states=True)
if SPARSITY != 1:
    load_masked_model_single(loaded_model, f'pruned_models/{model_name}-{SPARSITY}.pt')

t_optim = torch.optim.AdamW(params=loaded_model.parameters(), lr=1e-5)
loaded_model, t_optim, dataloader = accelerate.prepare(loaded_model, t_optim, dataloader)
for epoch in tqdm(range(EPOCH_COUNT)):
    training_step(loaded_model, training_data, t_optim)
torch.save(loaded_model.state_dict(), f'pruned_models/{model_name}-{SPARSITY}-finetuned.pt')