from os.path import exists
import torch
import torch.nn as nn
import nlp


# process the examples in input and target text format and the eos token at the end 
def add_eos_to_examples(example):
    example['input_text'] = 'question: %s  context: %s </s>' % (example['question'], example['context'])
    example['target_text'] = '%s </s>' % example['answers']['text'][0]
    return example

# tokenize the examples
def convert_to_features(example_batch):
    input_encodings = tokenizer.batch_encode_plus(example_batch['input_text'], pad_to_max_length=True, max_length=512)
    target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'], pad_to_max_length=True, max_length=16)

    encodings = {
        'input_ids': input_encodings['input_ids'], 
        'attention_mask': input_encodings['attention_mask'],
        'target_ids': target_encodings['input_ids'],
        'target_attention_mask': target_encodings['attention_mask']
    }

    return encodings

def create_or_load(tokenizer):
    if exists('data/train_data.pt') and xists('data/valid_data.pt'):
        train_dataset = torch.load('train_data.pt')
        valid_dataset = torch.load('valid_data.pt')
    else:
        tokenizer = tokenizer

        # load train and validation split of squad
        train_dataset  = nlp.load_dataset('squad', split=nlp.Split.TRAIN)
        valid_dataset = nlp.load_dataset('squad', split=nlp.Split.VALIDATION)

        # map add_eos_to_examples function to the dataset example wise 
        train_dataset = train_dataset.map(add_eos_to_examples)
        # map convert_to_features batch wise
        train_dataset = train_dataset.map(convert_to_features, batched=True)

        valid_dataset = valid_dataset.map(add_eos_to_examples, load_from_cache_file=False)
        valid_dataset = valid_dataset.map(convert_to_features, batched=True, load_from_cache_file=False)


        # set the tensor type and the columns which the dataset should return
        columns = ['input_ids', 'target_ids', 'attention_mask', 'target_attention_mask']
        train_dataset.set_format(type='torch', columns=columns)
        valid_dataset.set_format(type='torch', columns=columns)

        torch.save(train_dataset, 'data/train_data.pt')
        torch.save(valid_dataset, 'data/valid_data.pt')
    return train_dataset, valid_dataset

