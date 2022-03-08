from os.path import exists
import os
import json
import copy
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer

from dataclasses import dataclass, field
from transformers import (
    DataCollator,
)
from typing import Dict, List, Optional
import torch

#########################################################################################
#################### The Below Code Blocks will be for SQuAD & TextbookQA dataset #######
#########################################################################################

# process the examples in input and target text format and the eos token at the end
def add_eos_to_examples(example):
    example["input_text"] = "question: %s  context: %s </s>" % (
        example["question"],
        example["context"],
    )
    example["target_text"] = "%s </s>" % example["answers"][0] # always getting first answer
    return example


# tokenize the examples
def convert_to_features(example_batch):
    input_encodings = tokenizer.batch_encode_plus(
        example_batch["input_text"],
        pad_to_max_length=True,
        max_length=512,
        truncation=True,
    )
    target_encodings = tokenizer.batch_encode_plus(
        example_batch["target_text"],
        pad_to_max_length=True,
        max_length=16,
        truncation=True,
    )

    encodings = {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"],
        "target_ids": target_encodings["input_ids"],
        "target_attention_mask": target_encodings["attention_mask"],
    }

    return encodings

def load(dataset, partition, args):
    if partition == 'train':
        subset = args['train_set']
    elif partition == 'validation':
        subset = args['val_set']
    elif partition == 'test':
        subset = args['test_set']
    fp = "data/{}_{}.pt".format(partition, subset)
    if exists(fp):
        print(fp)
        data = torch.load(fp)
        columns = ["input_ids", "target_ids", "attention_mask", "target_attention_mask"]
        data.set_format(type="torch", columns=columns, format_kwargs=data.format["format_kwargs"])
    else:
        # preprocess data
        data = dataset[partition].filter(lambda example: example['subset']==subset)
        data = data.map(add_eos_to_examples)
        data = data.map(convert_to_features, batched=True)
        columns = ["input_ids", "target_ids", "attention_mask", "target_attention_mask"]
        data.set_format(type="torch", columns=columns, format_kwargs=data.format["format_kwargs"])
        torch.save(data, fp)
    return data


def get_dataset(input_tokenizer, args):
    global tokenizer
    tokenizer = input_tokenizer
    dataset = load_dataset("mrqa", cache_dir='data/mrqa')
    train_dataset = load(dataset, 'train', args)
    val_dataset = load(dataset, 'validation', args)
    test_dataset = load(dataset, 'test', args)
    return train_dataset, val_dataset, test_dataset

#########################################################################################
#################### The Above Code Blocks will be for SQuAD & TextbookQA dataset #######
#########################################################################################
