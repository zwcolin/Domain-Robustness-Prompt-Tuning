#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""


import argparse
import logging

import numpy as np
import torch
import json

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoConfig,
    set_seed,
    GPT2LMHeadModelAdapter,
)
import sys, os
from model import PrefixTuning


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PREFIX = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


# def set_seed(args):
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     if args.n_gpu > 0:
#         torch.cuda.manual_seed_all(args.seed)


#
# Functions to prepare models' input
#


def prepare_ctrl_input(args, _, tokenizer, prompt_text):
    if args.temperature > 0.7:
        logger.info(
            "CTRL typically works better with lower temperatures (and lower top_k)."
        )

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        logger.info(
            "WARNING! You are not starting your generation from a control code so you won't get good results"
        )
    return prompt_text


def prepare_xlm_input(args, model, tokenizer, prompt_text):
    # kwargs = {"language": None, "mask_token_id": None}

    # Set the language
    use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
    if hasattr(model.config, "lang2id") and use_lang_emb:
        available_languages = model.config.lang2id.keys()
        if args.xlm_language in available_languages:
            language = args.xlm_language
        else:
            language = None
            while language not in available_languages:
                language = input(
                    "Using XLM. Select language in "
                    + str(list(available_languages))
                    + " >>> "
                )

        model.config.lang_id = model.config.lang2id[language]
        # kwargs["language"] = tokenizer.lang2id[language]

    # TODO fix mask_token_id setup when configurations will be synchronized between models and tokenizers
    # XLM masked-language modeling (MLM) models need masked token
    # is_xlm_mlm = "mlm" in args.model_name_or_path
    # if is_xlm_mlm:
    #     kwargs["mask_token_id"] = tokenizer.mask_token_id

    return prompt_text


def prepare_xlnet_input(args, _, tokenizer, prompt_text):
    prefix = (
        args.prefix
        if args.prefix
        else args.padding_text
        if args.padding_text
        else PREFIX
    )
    prompt_text = prefix + prompt_text
    return prompt_text


def prepare_transfoxl_input(args, _, tokenizer, prompt_text):
    prefix = (
        args.prefix
        if args.prefix
        else args.padding_text
        if args.padding_text
        else PREFIX
    )
    prompt_text = prefix + prompt_text
    return prompt_text


PREPROCESSING_FUNCTIONS = {
    "ctrl": prepare_ctrl_input,
    "xlm": prepare_xlm_input,
    "xlnet": prepare_xlnet_input,
    "transfo-xl": prepare_transfoxl_input,
}


def read_e2e_files(path, tokenizer, lowdata_token=None):
    file_dict = {}
    with open(path, "r") as f:
        for line in f:
            src, tgt = line.strip().split("||")
            # URGENT CHANGE
            # src =  src + ' {}'.format(' summarize :')
            if lowdata_token is None:
                src = " {} {}".format(src, tokenizer.bos_token)
                # src =  src + ' {}'.format(tokenizer.bos_token)
            else:
                src = " {} {} {}".format(lowdata_token, src, tokenizer.bos_token)
            if src not in file_dict:
                file_dict[src] = []
            file_dict[src].append(tgt)
    return file_dict


def read_wp_files(path, tokenizer):
    file_dict = {}
    with open(path, "r") as f:
        for line in f:
            src, tgt = line.strip().split("|||")
            src = src + " {}".format(tokenizer.bos_token)
            if src not in file_dict:
                file_dict[src] = []
            file_dict[src].append(tgt)
    return file_dict


def read_classifySentiment_files(path, tokenizer):
    file_dict = []
    with open(path, "r") as f:
        for line in f:
            tgt, src = line.strip().split("|||")
            src = src.replace("< br / >", "\n")
            src = " {} {}".format(src, tokenizer.bos_token)
            file_dict.append((src, tgt))
    return file_dict


def read_classifyTopic_files(path, tokenizer):
    file_dict = []
    with open(path, "r") as f:
        for line in f:
            if len(line) > 0 and not line.isspace() and len(line.split("||")) == 2:
                tgt, src = line.strip().split("||")
            else:
                continue
            src = " {} {}".format(src, tokenizer.bos_token)
            file_dict.append((src, tgt))
    return file_dict


def read_sum_files(path, tokenizer, max_source_length, max_target_length):
    src_file = path
    tgt_file = path[:-6] + "target"

    file_dict = {}

    src_lines = []
    with open(src_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) > 0 and not line.isspace():
                src_lines.append(line)
    with open(tgt_file, encoding="utf-8") as f:
        tgt_lines = [
            line
            for line in f.read().splitlines()
            if (len(line) > 0 and not line.isspace())
        ]

    print(tgt_file, len(tgt_lines), "\n", src_file, len(src_lines))

    for src, tgt in zip(src_lines, tgt_lines):
        src_bpe = tokenizer.encode(
            src,
            add_special_tokens=False,
            truncation=True,
            max_length=max_source_length,
            is_split_into_words=False,
        )

        src_full = src_bpe + [tokenizer.bos_token_id]  # add the bos token.

        # print(len(src_full), src_full)
        src_full = tuple(src_full)
        if src_full not in file_dict:
            file_dict[src_full] = [tgt]
        else:
            print("should not happen")
            file_dict[src_full].append(tgt)
    return file_dict


def read_webnlg_files(path, tokenizer):
    file_dict = {}

    with open(path) as f:
        lines_dict = json.load(f)

    full_rela_lst = []
    full_src_lst = []
    # full_tgt_lst = []
    total_count = 0
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
            if i > 0:
                temp_triples += " | "
            temp_triples += "{} : {} : {}".format(subj, rela, obj)

        temp_triples = " {} {}".format(temp_triples, tokenizer.bos_token)

        for sent in sents:
            if True:  # sent["comment"] == 'good'
                if (temp_triples, tuple(rela_lst)) not in file_dict:
                    file_dict[(temp_triples, tuple(rela_lst))] = []
                    full_src_lst.append(temp_triples)
                    full_rela_lst.append(tuple(rela_lst))
                file_dict[(temp_triples, tuple(rela_lst))].append(sent["lex"])

    print(len(file_dict), len(full_src_lst))
    assert len(full_rela_lst) == len(full_src_lst)
    assert len(full_rela_lst) == len(file_dict)

    return file_dict


def read_triples_files2(path, tokenizer):
    file_src = []
    file_tgt = []

    with open(path) as f:
        lines_dict = json.load(f)

    print(len(lines_dict))
    full_rela_lst = []
    full_src_lst = []
    for example in lines_dict:
        rela_lst = []
        temp_triples = ""
        for i, tripleset in enumerate(example["tripleset"]):
            subj, rela, obj = tripleset
            rela = rela.lower()
            rela_lst.append(rela)
            if i > 0:
                temp_triples += " | "
            temp_triples += "{} : {} : {}".format(subj, rela, obj)

        temp_triples = " {} {}".format(temp_triples, tokenizer.bos_token)

        file_src.append((temp_triples, tuple(rela_lst)))
        # file_tgt

        for sent in example["annotations"]:
            if (temp_triples, tuple(rela_lst)) not in file_dict:
                file_dict[(temp_triples, tuple(rela_lst))] = []
                full_src_lst.append(temp_triples)
                full_rela_lst.append(tuple(rela_lst))
            file_dict[(temp_triples, tuple(rela_lst))].append(sent["text"])

    print(len(file_dict), len(full_src_lst))
    assert len(full_rela_lst) == len(full_src_lst)
    assert len(full_rela_lst) == len(file_dict)
    return file_dict


def read_triples_files(path, tokenizer):
    file_dict = {}

    with open(path) as f:
        lines_dict = json.load(f)

    print(len(lines_dict))
    full_rela_lst = []
    full_src_lst = []
    for example in lines_dict:
        rela_lst = []
        temp_triples = ""
        for i, tripleset in enumerate(example["tripleset"]):
            subj, rela, obj = tripleset
            rela = rela.lower()
            rela_lst.append(rela)
            if i > 0:
                temp_triples += " | "
            temp_triples += "{} : {} : {}".format(subj, rela, obj)

        temp_triples = " {} {}".format(temp_triples, tokenizer.bos_token)

        for sent in example["annotations"]:
            if (temp_triples, tuple(rela_lst)) not in file_dict:
                file_dict[(temp_triples, tuple(rela_lst))] = []
                full_src_lst.append(temp_triples)
                full_rela_lst.append(tuple(rela_lst))
            file_dict[(temp_triples, tuple(rela_lst))].append(sent["text"])

    print(len(file_dict), len(full_src_lst))
    assert len(full_rela_lst) == len(full_src_lst)
    assert len(full_rela_lst) == len(file_dict)
    return file_dict


# def write_e2e_corr(prompt_lst, file_dict, corr_path):
#     with open(corr_path, 'w') as f:
#         for x in prompt_lst:
#             for line in file_dict[x]:
#                 print(line, file=f)
#             print('', file=f)
#     return


def write_e2e_corr(prompt_lst, file_dict, corr_path):
    print(len(prompt_lst))
    with open(corr_path, 'w+',encoding="utf-8") as f:
        for x in prompt_lst:
            for line in file_dict[x]:
                if not line.strip():
                    print("PROBLEM", line, "PROBLEM", file_dict[x])
                else:
                    print(line, file=f)
            print("", file=f)

    # buf = [[]]
    # with open(corr_path, 'r') as fh:
    #     for line in fh:
    #         line = line.strip()
    #         if True:
    #             # print(line)
    #             if not line:
    #                 buf.append([])
    #             else:
    #                 buf[-1].append(line)
    #         else:
    #             buf.append(line)
    # if not buf[-1]:
    #     del buf[-1]
    #
    # print(buf[:3])
    #
    # print(len(buf))

    return


def write_e2e_src(prompt_lst, corr_path):
    with open(corr_path, "w") as f:
        for x in prompt_lst:
            print(x, file=f)
    return


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        required=False,
        help="Path to pre-trained tokenizer or shortcut name selected in the list: "
        + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument(
        "--prefixModel_name_or_path",
        default=None,
        type=str,
        required=False,
        help="Path to pre-trained PrefixTuning Model or shortcut name selected in the list: "
        + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--task_mode", type=str, default="embMatch")
    parser.add_argument("--control_mode", type=str, default="yes")
    parser.add_argument("--prefix_mode", type=str, default="activation")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--gen_dir", type=str, default="e2e_results_conv")
    parser.add_argument(
        "--stop_token",
        type=str,
        default=None,
        help="Token at which text generation is stopped",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="primarily useful for CTRL model; in that case, use 1.2",
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)

    parser.add_argument(
        "--tuning_mode", type=str, default="finetune", help="prefixtune or finetune"
    )
    parser.add_argument("--eval_dataset", type=str, default="val", help="val or test")
    parser.add_argument("--objective_mode", type=int, default=2)
    parser.add_argument(
        "--format_mode", type=str, default="peek", help="peek, cat, nopeek, or infix"
    )
    parser.add_argument("--optim_prefix", type=str, default="no", help="optim_prefix")
    parser.add_argument("--preseqlen", type=int, default=5, help="preseqlen")

    parser.add_argument(
        "--prefix", type=str, default="", help="Text added prior to input."
    )
    parser.add_argument(
        "--control_dataless", type=str, default="no", help="control dataless mode"
    )
    parser.add_argument(
        "--padding_text",
        type=str,
        default="",
        help="Deprecated, the use of `--prefix` is preferred.",
    )
    parser.add_argument(
        "--xlm_language",
        type=str,
        default="",
        help="Optional language when used with the XLM model.",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="The number of samples to generate.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    args = parser.parse_args()

    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    logger.warning(
        "device: %s, n_gpu: %s, 16-bits training: %s",
        args.device,
        args.n_gpu,
        args.fp16,
    )

    set_seed(args.seed)

    # Initialize the model and tokenizer

    if args.tuning_mode == "prefixtune":

        print(
            "loading from PrefixTuning.",
            args.prefixModel_name_or_path,
        )
        if args.model_name_or_path:
            config = AutoConfig.from_pretrained(
                args.model_name_or_path, cache_dir=args.cache_dir
            )
        else:
            assert False, "shouldn not init config from scratch. "
            config = CONFIG_MAPPING[args.model_type]()
            logger.warning("You are instantiating a new config instance from scratch.")

        try:
            args.model_type = args.model_type.lower()
            model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        except KeyError:
            raise KeyError(
                "the model {} you specified is not supported. You are welcome to add it and open a PR :)"
            )

        if args.model_name_or_path:
            print("loading the trained tokenizer")
            tokenizer = tokenizer_class.from_pretrained(
                args.model_name_or_path, cache_dir=args.cache_dir
            )
        elif args.tokenizer_name:
            print("loading from the init tokenizer")
            tokenizer = tokenizer_class.from_pretrained(
                args.tokenizer_name, cache_dir=args.cache_dir
            )

        # TODAYFIX.
        config._my_arg_tune_mode = args.tuning_mode
        config._my_arg_task_mode = args.task_mode
        config._objective_mode = args.objective_mode
        model = model_class.from_pretrained(
            args.model_name_or_path, config=config, cache_dir=args.cache_dir
        )
        model.to(args.device)

        print(
            len(tokenizer),
            tokenizer.bos_token,
            tokenizer.eos_token,
            tokenizer.pad_token,
        )

        # TODO LISA
        add_pad = False

        if args.model_name_or_path == "gpt2-medium":
            if args.task_mode == "dataless":
                print(args.tuning_mode, "dataless setting, so no new tokens at all.")
                print(
                    "We do not add special tokens to the tokenizer, instead, we just finetune on <|endoftext|>"
                )

                print(tokenizer.eos_token_id)
                print(tokenizer.eos_token)
                print(tokenizer.pad_token_id)
                tokenizer.pad_token = tokenizer.eos_token
                print(tokenizer.pad_token, tokenizer.pad_token_id)

            elif add_pad:
                print("extending the size of word embeddings. to include the [PAD] ")
                num_added_tokens = tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                embedding_layer = model.resize_token_embeddings(len(tokenizer))
            else:
                print(tokenizer.eos_token_id)
                print(tokenizer.eos_token)
                print(tokenizer.pad_token_id)
                tokenizer.pad_token = tokenizer.eos_token
                print(tokenizer.pad_token, tokenizer.pad_token_id)

            ########################################3

        print(
            len(tokenizer),
            tokenizer.bos_token,
            tokenizer.eos_token,
            tokenizer.pad_token,
        )

        gpt2 = model

        # config._my_arg_task_mode = args.task_mode
        # config._my_arg_control = True
        # config.train_weights = 'no'
        print(config)
        if args.optim_prefix == "yes":
            optim_prefix_bool = True
        elif args.optim_prefix == "no":
            optim_prefix_bool = False
        else:
            assert False, "model_args.optim_prefix should be either yes or no"

        if args.prefixModel_name_or_path is not None:

            #################
            #
            config = AutoConfig.from_pretrained(
                args.prefixModel_name_or_path, cache_dir=args.cache_dir
            )
            print(config)

            if args.prefix_mode == "activation":
                model = PrefixTuning.from_pretrained(
                    args.prefixModel_name_or_path,
                    from_tf=bool(
                        ".ckpt" in args.prefixModel_name_or_path,
                    ),
                    config=config,
                    model_gpt2=gpt2,
                    optim_prefix=optim_prefix_bool,
                    preseqlen=args.preseqlen,
                    use_infix=(args.format_mode == "infix"),
                )
            #
            ######################

            # model = PrefixTuning.from_pretrained(
            #     args.prefixModel_name_or_path,
            #     from_tf=bool(".ckpt" in args.prefixModel_name_or_path,),
            #     config=config,
            #     model_gpt2=gpt2, optim_prefix=optim_prefix_bool, preseqlen=args.preseqlen,
            # )
            model.to(args.device)

            # print('-'*100)
            # print(model.training)
            # print(gpt2.training)
            # model.train()
            # gpt2.train()
            # print(model.training)
            # print(gpt2.training)
            # model.eval()
            # gpt2.eval()
            # print(model.training)
            # print(gpt2.training)
            # print('-' * 100)

        else:
            assert False, "prefixModel_name_or_path is NONE."

    if args.fp16:
        model.half()

    args.length = adjust_length_to_model(
        args.length, max_sequence_length=model.config.max_position_embeddings
    )
    logger.info(args)

    if args.task_mode == "webnlg":
        QUICK_CHECK = False
        if args.task_mode == "webnlg":
            if args.eval_dataset == "valid":
                test_path = (
                    "/home/l6wang/PrefixTuning/data/webnlg_challenge_2017/train.json"
                )
            elif args.eval_dataset == "test":
                test_path = (
                    "/home/l6wang/PrefixTuning/data/webnlg_challenge_2017/test.json"
                )
            else:
                assert False, "eval_dataset needs to be [valid, test]"
            prompt_text_dict = read_webnlg_files(test_path, tokenizer)

        if QUICK_CHECK:
            prompt_text_pair = list(prompt_text_dict.keys())[:20]
            prompt_text_lst, prompt_rela_lst = zip(*prompt_text_pair)
            decode_mode = "beam"

        else:
            prompt_text_pair = list(prompt_text_dict.keys())
            prompt_text_lst, prompt_rela_lst = zip(*prompt_text_pair)
            if args.prefixModel_name_or_path is not None:
                temp = os.path.basename(args.prefixModel_name_or_path)
            else:
                temp = os.path.basename(args.model_name_or_path)
            split_file = args.eval_dataset  # test
            decode_mode = "beam"
            curr_dir = os.path.join(
                "./res/", args.gen_dir, "{}_{}_{}".format(temp, split_file, decode_mode)
            )
            print(curr_dir)
            gold_dir = os.path.join(
                "./res/", args.gen_dir, "{}_{}_{}".format(temp, split_file, "gold")
            )
            print(gold_dir)
            write_e2e_corr(prompt_text_pair, prompt_text_dict, gold_dir)
            src_dir = os.path.join(
                "./res/", args.gen_dir, "{}_{}_{}".format(temp, split_file, "src")
            )
            write_e2e_src(prompt_text_pair, src_dir)

            out_handle = open(curr_dir, "w")

    for prompt_idx, prompt_text in enumerate(prompt_text_lst):

        # Different models need different input formatting and/or extra arguments
        requires_preprocessing = args.model_type in PREPROCESSING_FUNCTIONS.keys()
        if requires_preprocessing:
            prepare_input = PREPROCESSING_FUNCTIONS.get(args.model_type)
            preprocessed_prompt_text = prepare_input(
                args, model, tokenizer, prompt_text
            )

            if model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
                tokenizer_kwargs = {"add_space_before_punct_symbol": True}
            else:
                tokenizer_kwargs = {}

            encoded_prompt = tokenizer.encode(
                preprocessed_prompt_text,
                add_special_tokens=False,
                return_tensors="pt",
                **tokenizer_kwargs
            )
        else:
            prefix = args.prefix if args.prefix else args.padding_text
            encoded_prompt = tokenizer.encode(
                prefix + prompt_text, add_special_tokens=False, return_tensors="pt"
            )
        encoded_prompt = encoded_prompt.to(args.device)

        if encoded_prompt.size()[-1] == 0:
            input_ids = None
        else:
            input_ids = encoded_prompt

        if args.control_mode == "yes" and args.control_dataless != "yes":
            # URGENT, check whether the next line is necessary?
            # control_code = torch.LongTensor(control_codes[prompt_idx]).to(model.device).unsqueeze(0).expand(args.num_return_sequences, -1)
            control_code = None
            pass
        else:
            control_code = None
        # for param in model.base_model.parameters():
        #     print(param.requires_grad)

        # if args.control_dataless == 'yes':
        if args.tuning_mode == "prefixtune":
            if args.task_mode == "webnlg":
                src = prompt_text_lst[prompt_idx].split()[:-1]
                print(src)
                src = " ".join(src)
                cat = prompt_rela_lst[prompt_idx]
                print(cat)
                src_cat = tokenizer(
                    cat,
                    add_special_tokens=True,
                    truncation=True,
                    is_split_into_words=True,
                )["input_ids"]
                src = tokenizer(
                    src,
                    add_special_tokens=True,
                    truncation=True,
                    is_split_into_words=False,
                )["input_ids"]

                mode = "cat"
                print(mode)

                if mode == "cat":
                    cc = src_cat

                # print('control code is ', cc)

                control_code = torch.LongTensor(cc).to(model.device).unsqueeze(0)


                # TODO.LISA
                if config.optim_prefix:
                    control_code = None

            else:
                control_code = None
                print("control code is None")

            if args.format_mode != "infix":
                print(config.optim_prefix, optim_prefix_bool)
                print("control code is ", control_code)
                prompt = model.get_prompt(control_code, gpt2=gpt2, bsz=1)
            else:
                print(control_code)
                print(src)
                src = torch.LongTensor(src).to(model.device).unsqueeze(0)
                print(input_ids)
                prompt = model.get_prompt(
                    src, None, gpt2=gpt2, bsz=1
                )  # src, control_code=None, gpt2=None, bsz=None, attn_mask=None

            # if args.task_mode == 'writingPrompts':
            #     prompt = None
            # else:
            # print(args.num_return_sequences)
            prompt = [
                x.expand(-1, args.num_return_sequences, -1, -1, -1) for x in prompt
            ]
            # print(prompt[0].shape)
            # print(input_ids.shape)

            # assert control_code is None
            print(decode_mode)
            if decode_mode == "nucleus":
                output_sequences = gpt2.generate(
                    input_ids=input_ids,
                    emb_match=None,
                    control_code=None,
                    past_key_values=prompt,
                    max_length=args.length + len(encoded_prompt[0]),
                    temperature=args.temperature,
                    top_k=args.k,
                    top_p=0.8,
                    repetition_penalty=args.repetition_penalty,
                    do_sample=True,
                    num_return_sequences=args.num_return_sequences,
                )
            elif decode_mode == "beam":
                ############################
                # torch.set_printoptions(profile="full")
                # print(input_ids)
                # print()
                # torch.set_printoptions(profile="default")
                # print(prompt[5][0][0][0])

                #############################
                output_sequences = gpt2.generate(
                    input_ids=input_ids,
                    emb_match=None,
                    control_code=None,
                    past_key_values=prompt,
                    max_length=args.length + len(encoded_prompt[0]),
                    min_length=5,
                    temperature=args.temperature,
                    top_k=args.k,
                    top_p=0.9,  # top_p=0.5,
                    repetition_penalty=args.repetition_penalty,  ##args.repetition_penalty,
                    do_sample=False,
                    num_beams=5,
                    bad_words_ids=[[628], [198]] if True else None,
                    num_return_sequences=1,
                )
                # print(output_sequences)

            elif decode_mode == "greedy":
                output_sequences = gpt2.generate(
                    input_ids=input_ids,
                    emb_match=None,
                    control_code=None,
                    past_key_values=prompt,
                    max_length=args.length + len(encoded_prompt[0]),
                    min_length=5,
                    temperature=args.temperature,
                    top_k=args.k,
                    top_p=0.5,
                    repetition_penalty=args.repetition_penalty,
                    do_sample=False,
                    bad_words_ids=[[628], [198]] if True else None,
                    num_return_sequences=1,
                )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences = []

        if QUICK_CHECK:
            for generated_sequence_idx, generated_sequence in enumerate(
                output_sequences
            ):
                print(
                    "=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1)
                )
                # args.stop_token = tokenizer.eos_token
                generated_sequence = generated_sequence.tolist()

                # Decode text
                text = tokenizer.decode(
                    generated_sequence, clean_up_tokenization_spaces=True
                )

                # Remove all text after the stop token
                text = text[: text.find(args.stop_token) if args.stop_token else None]

                # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
                total_sequence = (
                    prompt_text
                    + text[
                        len(
                            tokenizer.decode(
                                encoded_prompt[0], clean_up_tokenization_spaces=True
                            )
                        ) :
                    ]
                )

                generated_sequences.append(total_sequence)
                print(total_sequence)
        else:
            for generated_sequence_idx, generated_sequence in enumerate(
                output_sequences
            ):
                print(
                    "=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1)
                )
                # args.stop_token = tokenizer.eos_token
                generated_sequence = generated_sequence.tolist()

                # Decode text
                text = tokenizer.decode(
                    generated_sequence, clean_up_tokenization_spaces=True
                )

                print(text)
                text_output = text[
                    len(
                        tokenizer.decode(
                            encoded_prompt[0], clean_up_tokenization_spaces=True
                        )
                    ) :
                ]
                idx = text_output.find(tokenizer.eos_token)
                if idx >= 0:
                    text_output = text_output[:idx]
                text_output = text_output.strip()

                if args.task_mode == "topic" or args.task_mode == "sentiment":
                    text_output = prompt_text + " " + text_output + " [SPECIAL_END]"

                if text_output:
                    print(text_output, file=out_handle)
                else:
                    print("Error", file=out_handle)

        print()

    # return generated_sequences

    if not QUICK_CHECK:
        out_handle.close()

    if args.task_mode == "webnlg":
        out_file_eval = curr_dir + "_eval"
        print(out_file_eval, "\n", gold_dir, "\n", curr_dir)
        tagging = os.path.basename(curr_dir)
        # Need to download from https://github.com/Yale-LILY/dart/blob/master/evaluation/run_eval_on_webnlg.sh
        os.system(
            "bash run_eval_on_webnlg.sh "
            "{} {} >> {}".format(curr_dir, tagging, out_file_eval)
        )


if __name__ == "__main__":
    main()
