#imports
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, OPTForCausalLM
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

#hyperparam test, remove later
EPOCH_COUNT = 5

#set device
device = 'cuda' if torch.cuda.is_available() else 'cpu' 


def fine_tune(model, training_data, EPOCH_COUNT, tokenizer):
    
    #dataloader from dataloader (mlm=False when training without mask)
    reformatted_data = DataLoader(training_data, collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False))

    t_optim = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
    for epoch in range(EPOCH_COUNT):
        training_data.set_epoch(epoch)
        for i, batch in enumerate(tqdm(reformatted_data, total=5)):
            if i == 5:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs[0]
            loss.backward()
            t_optim.step()
            t_optim.zero_grad()
            if i % 10 == 0:
                print(f"loss: {loss}")


# if __name__ == '__main__':
#     #stream c4, training split
#     training_data = load_dataset('c4', "en", split='train', streaming=True)

#     #load tokenizer
#     tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')

#     #IMPORTANT: process data while streaming -> remove unnecessary columns in batches
#     training_data = training_data.map(encode_tok, batched=True, remove_columns=["text", "timestamp", "url"])

#     #set data to tensor mode
#     training_data = training_data.with_format("torch")

#       OR replace everything above with:
    #encode tokens
    # def encode_tok(examples):
    #     return tokenizer(examples['text'], truncation=True, padding='max_length')

    # training_data = training_data.map(encode_tok, batched=True, remove_columns=["text", "timestamp", "url"])
    # training_data = training_data.with_format("torch")


#     #dataloader from dataloader (mlm=False when training without mask)
#     reformatted_data = DataLoader(training_data, collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False))

#     # loading model for training
#     model = OPTForCausalLM.from_pretrained("facebook/opt-125m")

#     #training loop
#     model.train().to(device)