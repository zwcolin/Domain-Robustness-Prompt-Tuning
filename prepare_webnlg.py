from os.path import exists
import os
import json
import copy
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer

from dataclasses import dataclass, field
from transformers import (
    DataCollator,
)
from typing import Dict, List, Optional
import torch


class LineByLineWebNLGTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        bos_tok: str,
        eos_tok: str,
    ):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        #       logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path) as f:
            lines_dict = json.load(f)

        full_rela_lst = []
        full_src_lst = []
        full_tgt_lst = []

        for i, example in enumerate(lines_dict["entries"]):
            sents = example[str(i + 1)]["lexicalisations"]
            triples = example[str(i + 1)]["modifiedtripleset"]

            rela_lst = []
            temp_triples = ""
            for j, tripleset in enumerate(triples):
                subj, rela, obj = (
                    tripleset["subject"],
                    tripleset["property"],
                    tripleset["object"],
                )
                rela_lst.append(rela)
                temp_triples += " | "
                temp_triples += "{} : {} : {}".format(subj, rela, obj)

            for sent in sents:
                if sent["comment"] == "good":
                    full_tgt_lst.append(sent["lex"])
                    full_src_lst.append(temp_triples)
                    full_rela_lst.append(rela_lst)

        assert len(full_rela_lst) == len(full_src_lst)
        assert len(full_rela_lst) == len(full_tgt_lst)

        edited_sents = []
        for src, tgt in zip(full_src_lst, full_tgt_lst):
            sent = " {} {} ".format(src, bos_tok) + tgt + " {}".format(eos_tok)
            edited_sents.append(sent)

        batch_encoding = tokenizer(
            edited_sents,
            add_special_tokens=True,
            truncation=True,
            max_length=block_size,
            is_split_into_words=False,
        )
        self.examples = batch_encoding["input_ids"]

        self.labels = copy.deepcopy(self.examples)

        # split into category words:
        ssl_lst = full_rela_lst

        self.src_cat = tokenizer(
            ssl_lst,
            add_special_tokens=True,
            truncation=True,
            max_length=block_size,
            is_split_into_words=True,
        )["input_ids"]

        self.src_sent = []
        self.tgt_sent = []
        temp_src_len = 0
        temp_tgt_len = 0
        temp_count = 0

        if True:
            separator = tokenizer(bos_tok, add_special_tokens=False)["input_ids"][0]
            for i, elem in enumerate(self.labels):
                sep_idx = elem.index(separator) + 1
                self.src_sent.append(
                    self.examples[i][: sep_idx - 1]
                )  # does not contain the BOS separator
                self.tgt_sent.append(
                    self.examples[i][sep_idx - 1 :]
                )  # contains the BOS separator.
                self.labels[i][:sep_idx] = [-100] * sep_idx
                temp_src_len += sep_idx - 1
                temp_tgt_len += len(elem) - (sep_idx - 1)
                temp_count += 1

        #         print('tgt_avg: ', temp_tgt_len / temp_count)
        #         print('src_avg: ', temp_src_len / temp_count)
        #         print('ratios: ', temp_src_len / temp_tgt_len)

        #         print(self.labels[0])
        #         print(self.examples[0])
        #         print(edited_sents[0])
        #         print(self.src_sent[0])
        #         print(self.tgt_sent[0])
        #         print(self.src_cat[0])
        #         print()
        #         print(self.labels[1])
        #         print(self.examples[1])
        #         print(edited_sents[1])
        #         print(self.src_sent[1])
        #         print(self.tgt_sent[1])
        #         print(self.src_cat[1])
        assert len(self.src_cat) == len(self.examples)

    def __len__(self):
        return len(self.examples)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        return (
            torch.tensor(self.examples[i], dtype=torch.long),
            torch.tensor(self.labels[i], dtype=torch.long),
            torch.tensor(self.src_sent[i], dtype=torch.long),
            torch.tensor(self.tgt_sent[i], dtype=torch.long),
            torch.tensor(self.src_cat[i], dtype=torch.long),
        )


def get_dataset(tokenizer: PreTrainedTokenizer, file_path: str):
    dataset = LineByLineWebNLGTextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=1024,
        bos_tok=tokenizer.bos_token,
        eos_tok=tokenizer.eos_token,
    )
    return dataset
