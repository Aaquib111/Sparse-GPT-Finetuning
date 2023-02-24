# SparseGPT fine tune loop
from SparseGPT_pruning import sparsegpt_prune
from trainingv2 import fine_tune
from save_pruned_model import mask_from_pruned, unmask_model

# first, calibrate model
model = OPTForCausalLM.from_pretrained(model_name, output_attentions=True,
                                        output_hidden_states=True).to(device=device) # type: ignore
feature_hessians = {}
#put_input_hooks(model=model, features=feature_hessians, storage_dir=storage_dir, offload_freq=10000, feature_storage_device='cpu')
put_input_hooks(model=model, features=feature_hessians, feature_storage_device='cpu')
split_model_calibration(model)
torch.cuda.empty_cache()


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

def get_prop_zeros(model):
    return torch.sum(model.get_decoder().layers[0].self_attn.k_proj.weight == 0) / (torch.numel(model.get_decoder().layers[0].self_attn.k_proj.weight))

# now, fine tune loop
# sparseness is defined as proportion of nonzeros (opposite of intuitive)
# sparseness_sequence = [.9, .8, .7, .6, .5, .4, .3, .2]
sparseness_sequence = [.9, .5, .4]

for sparseness_index in range(len(sparseness_sequence)):

    if sparseness_index == 0:
        sparseness_dif = sparseness_sequence[sparseness_index]
    else:
        sparseness_dif = sparseness_sequence[sparseness_index] / sparseness_sequence[sparseness_index -1]
    
    sparsegpt_prune(model=model, feature_hessians=feature_hessians, SPARSENESS=sparseness_dif, EPSILON=EPSILON, B=B, Bs=Bs)
    print(f"After pruning, Model has {get_prop_zeros(model)}")

    # activate masks
    mask_from_pruned(model=model)

    # fine_tune(model=model, training_data=training_data, EPOCH_COUNT=1, tokenizer=tokenizer)
    
    print(f"After fine-tuning, Model has {get_prop_zeros(model)}")

    # deactivate masks
    unmask_model(model=model)
    for n, m in model.named_buffers():
        print(n)

    pruned_model_name = f'{model_size}-test-{sparseness_sequence[sparseness_index]}'
    torch.save(model.state_dict(), f'pruned_models/{pruned_model_name}.pt')