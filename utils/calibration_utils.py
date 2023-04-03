# run model on batches of calibration data, then concatenate inputs
def split_model_calibration(model, dataset):
    batch_sentences = []
    for i, data in tqdm(enumerate(iter(dataset['train'])), total=calibration_size):
        if i < calibration_size + 1:
            if len(batch_sentences) >= calibration_batch_size:
                with torch.no_grad():
                    encoded_input = tokenizer(batch_sentences, return_tensors="pt",
                                              padding="max_length", max_length=token_length,
                                              truncation=True).to(device=device)
                    model(**encoded_input, labels=encoded_input.input_ids)
                    del encoded_input
                    torch.cuda.empty_cache()
                    batch_sentences = []
            batch_sentences.append(data['text'])
        else:
            break