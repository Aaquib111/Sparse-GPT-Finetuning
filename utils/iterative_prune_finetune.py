# SparseGPT fine tune loop
from utils.prune_utils import sparsegpt_prune
from cerebras_pruning import mask_lowest
from trainingv2 import fine_tune
from utils.save_utils import mask_from_pruned, unmask_model
import torch

def get_prop_zeros(model):
    return torch.sum(model.module.get_decoder().layers[0].self_attn.k_proj.weight == 0) / (torch.numel(model.module.get_decoder().layers[0].self_attn.k_proj.weight))

# now, fine tune loop
# sparseness is defined as proportion of nonzeros (opposite of intuitive)
# sparseness_sequence = [.9, .8, .7, .6, .5, .4, .3, .2]
# sparseness_sequence = [.9, .5, .4]

# model is model to iteratively tune and prune, will do so in place
# model_size is for naming the save files (like opt-125m)
# sparseness sequence is sequence of sparsities (.8 sparseness = 20% proportion of zeros)
# training_data and tokenizer are for fine_tuning (should already be preprocessed with torch.format and stuff)
def iterative_sparsegpt_prune_tune(model, model_size, sparseness_sequence, feature_hessians, EPSILON, B, Bs, tokenizer, EPOCH_COUNT):
    # for sparseness_index in range(len(sparseness_sequence)):
    for sparseness in sparseness_sequence:
        sparsegpt_prune(model=model, model_name=model_size,
                        feature_hessians=feature_hessians,
                        SPARSENESS=sparseness, EPSILON=EPSILON,
                        B=B, Bs=Bs, save_model=False)
        torch.cuda.empty_cache()
        fine_tune(model=model, EPOCH_COUNT=EPOCH_COUNT, tokenizer=tokenizer)

        # deactivate masks
        unmask_model(model=model)
        torch.cuda.empty_cache()
        print(f"proportion of zeros: {get_prop_zeros(model)}")

        pruned_model_name = f'{model_size}-finetuned-{sparseness}'
        torch.save(model.state_dict(), f'pruned_models/{pruned_model_name}-iterative.pt')

def iterative_cerebras_prune_tune(model, model_size, sparseness_sequence, training_data, tokenizer, EPOCH_COUNT):
    for sparseness in sparseness_sequence:

        mask_lowest(model=model, amount=1-sparseness)

        # activate masks
        mask_from_pruned(model=model)

        fine_tune(model=model, training_data=training_data, EPOCH_COUNT=EPOCH_COUNT, tokenizer=tokenizer)

        # deactivate masks
        unmask_model(model=model)

        print(f"proportion of zeros: {get_prop_zeros(model)}")

        pruned_model_name = f'{model_size}-cerebras-tune-and-prune-{sparseness}'
        torch.save(model.state_dict(), f'pruned_models/{pruned_model_name}.pt')
