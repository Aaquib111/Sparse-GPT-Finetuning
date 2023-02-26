# SparseGPT fine tune loop
from SparseGPT_pruning import sparsegpt_prune
from cerebras_pruning import mask_lowest
from trainingv2 import fine_tune
from save_pruned_model import mask_from_pruned, unmask_model
import torch
import gc

def get_prop_zeros(model):
    return torch.sum(model.get_decoder().layers[0].self_attn.k_proj.weight == 0) / (torch.numel(model.get_decoder().layers[0].self_attn.k_proj.weight))

# now, fine tune loop
# sparseness is defined as proportion of nonzeros (opposite of intuitive)
# sparseness_sequence = [.9, .8, .7, .6, .5, .4, .3, .2]
# sparseness_sequence = [.9, .5, .4]

# model is model to iteratively tune and prune, will do so in place
# model_size is for naming the save files (like opt-125m)
# sparseness sequence is sequence of sparsities (.8 sparseness = 20% proportion of zeros)
# training_data and tokenizer are for fine_tuning (should already be preprocessed with torch.format and stuff)
def iterative_sparsegpt_prune_tune(model, model_size, sparseness_sequence, feature_hessians, EPSILON, B, Bs, training_data, tokenizer, EPOCH_COUNT):
    # for sparseness_index in range(len(sparseness_sequence)):
    for sparseness in sparseness_sequence:

        # print(f"memory available before prune: {torch.cuda.memory_allocated(0)}")
        sparsegpt_prune(model=model, feature_hessians=feature_hessians, SPARSENESS=sparseness, EPSILON=EPSILON, B=B, Bs=Bs)
        print(f"memory available after prune: {torch.cuda.memory_allocated(0)}")

        # activate masks
        mask_from_pruned(model=model)
        print(f"memory allocated after generating masks: {torch.cuda.memory_allocated(0)}")
        fine_tune(model=model, training_data=training_data, EPOCH_COUNT=EPOCH_COUNT, tokenizer=tokenizer)

        # deactivate masks
        unmask_model(model=model)
        print(f"memory available after fine tuning: {torch.cuda.memory_allocated(0)}")

        print(f"proportion of zeros: {get_prop_zeros(model)}")
        print("bye")
        print(f"memory summary: {torch.cuda.memory_summary()}")
        print("hi")

        pruned_model_name = f'{model_size}-sparsegpt-tune-and-prune-{sparseness}'
        torch.save(model.state_dict(), f'pruned_models/{pruned_model_name}.pt')
        torch.cuda.empty_cache()
        gc.collect()



def iterative_cerebras_prune_tune(model, model_size, sparseness_sequence, training_data, dataloader, tokenizer, EPOCH_COUNT):
    for sparseness in sparseness_sequence:
        print(f"memory available before prune: {torch.cuda.memory_allocated(0)}")
        mask_lowest(model=model, amount=1-sparseness)
        print(f"memory available after prune: {torch.cuda.memory_allocated(0)}")

        # activate masks
        mask_from_pruned(model=model)

        fine_tune(model=model, training_data=training_data, reformatted_data = dataloader, EPOCH_COUNT=EPOCH_COUNT, tokenizer=tokenizer)

        # deactivate masks
        unmask_model(model=model)
        print(f"memory available after fine tuning: {torch.cuda.memory_allocated(0)}")

        print(f"proportion of zeros: {get_prop_zeros(model)}")
        print("bye")
        print(f"memory summary: {torch.cuda.memory_summary()}")
        print("hi")

        pruned_model_name = f'{model_size}-cerebras-tune-and-prune-{sparseness}'
        torch.save(model.state_dict(), f'pruned_models/{pruned_model_name}.pt')
        torch.cuda.empty_cache()
        gc.collect()
