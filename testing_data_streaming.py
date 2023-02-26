import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, DataCollatorForLanguageModeling
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer, OPTForCausalLM, pipeline
# dataset = load_dataset("mc4", "en", streaming=True, split="train")
# tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
# def encode(examples):
#     return tokenizer(examples['text'], truncation=True, padding='max_length')
# dataset = dataset.map(encode, batched=True, remove_columns=["text", "timestamp", "url"])

# dataset = dataset.with_format("torch")
# dataloader = DataLoader(dataset, collate_fn=DataCollatorForLanguageModeling(tokenizer))
# device = 'cuda' if torch.cuda.is_available() else 'cpu' 
# model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = OPTForCausalLM.from_pretrained('facebook/opt-125m', output_attentions=True,
                                        output_hidden_states=True).to(device=device) # type: ignore
# set up data
def encode_tok(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length')
#stream c4, training split
training_data = load_dataset('wikitext', "wikitext-2-raw-v1", split='train', streaming=True)
#load tokenizer
tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')
#IMPORTANT: process data while streaming -> remove unnecessary columns in batches
training_data = training_data.map(encode_tok, batched=True, remove_columns=["text"])
#set data to tensor mode
training_data = training_data.with_format("torch")

#set data to tensor mode
dataloader = DataLoader(training_data, collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False))

print(f"after epoch, allocated memory: {torch.cuda.memory_allocated(0)}")

model.train().to(device)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
for epoch in range(3):
    training_data.set_epoch(epoch)
    for i, batch in enumerate(dataloader):
        if i == 5:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 10 == 0:
            print(f"loss: {loss}")
    print(f"after epoch, allocated memory: {torch.cuda.memory_allocated(0)}")