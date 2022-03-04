import json, os, sys
import torch
from datetime import datetime
from textwrap import wrap
from transformers.optimization import Adafactor
from transformers import (
    T5Tokenizer,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    # DataCollatorForData2TextLanguageModeling
)
from model_prompt_tuning import T5PromptTuningLM, GPT2PromptTuningLM
from utils_prompt_tuning import evaluate
from prepare_data import get_dataset as get_dataset_qa
from prepare_webnlg import get_dataset as get_dataset_t2t
from collator import T2TDataCollator, DataCollatorForData2TextLanguageModeling
from run_generation import read_webnlg_files, read_triples_files

os.environ["WANDB_DISABLED"] = "true"

class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)

def setup_logs(args):
    if args['logging']:
        print('File logging is enabled')
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        path = "{}/{}/{}/{}/{}".format(args["method"], args["train_set"], args["model"], args["n_tokens"], timestamp)
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        sys.stderr = open(os.path.join(path, "err.txt"), "w")
        sys.stdout = open(os.path.join(path, "log.txt"), "w")
        args['path'] = path
        return path

def get_data_model(args):
    if "t5" in args["model"]:
        tokenizer = T5Tokenizer.from_pretrained(args["model"])
        model = T5PromptTuningLM.from_pretrained(
            args["model"],
            n_tokens=args["n_tokens"],
            soft_prompt_path=args["soft_prompt_path"],
            initialize_from_vocab=args["initialize_from_vocab"],
            random_range=args["random_range"],
        )
    if "gpt" in args["model"]:
        tokenizer = GPT2Tokenizer.from_pretrained(args["model"])
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2PromptTuningLM.from_pretrained(
            args["model"],
            n_tokens=args["n_tokens"],
            soft_prompt_path=args["soft_prompt_path"],
            initialize_from_vocab=args["initialize_from_vocab"],
            random_range=args["random_range"],
        )
        model.resize_token_embeddings(len(tokenizer))
    print(args)
    if args['task'] == 'qa':
        train_set, val_set, test_set = get_dataset_qa(tokenizer, args)
    if args['task'] == 't2t':
        train_set = get_dataset_t2t(
            tokenizer=tokenizer,
            file_path="./prefix_data/webnlg_challenge_2017/train.json",
        )
        val_set = get_dataset_t2t(
            tokenizer=tokenizer,
            file_path="./prefix_data/webnlg_challenge_2017/dev.json",
        )
        test_set = None

    return {
        'model': model,
        'tokenizer': tokenizer,
        'train_set': train_set,
        'val_set': val_set,
        'test_set': test_set,
    }

def get_optim(model, args):
    if args['optimizer'] == 'adafactor':
        optimizer = Adafactor(model.parameters(), 
            scale_parameter=args['scale_parameter'], 
            relative_step=args['relative_step'], 
            warmup_init=args['warmup_init'], 
            lr=args['lr'],
            clip_threshold=args['clip_threshold'])
        # TODO: we might need s scheduler
        lr_scheduler = None
    else:
        pass
    return {
        'optim': optimizer,
        'scheduler': lr_scheduler,
    }

def get_training_args(args):
    if args['logging']:
        args['artifact_dir'] = args['path'] + '/artifact'
    else:
        args['artifact_dir'] = './tmp'
    training_args = TrainingArguments(
            remove_unused_columns=False,
            per_device_train_batch_size=args['bz'],  # batch size per device during training
            per_device_eval_batch_size=args['bz'],   # batch size for evaluation
            num_train_epochs=args['epoch'],
            disable_tqdm=args['logging'],
            report_to=None,
            output_dir=args['artifact_dir'],          # output directory
            save_strategy='epoch',
            save_total_limit=1,

            logging_dir=args['path'],            # directory for storing logs
            logging_steps=20,
            
        )
    return training_args

def generate_predictions(model, tokenizer, dataset, debug):
    predictions = {}
    length = 10 if debug else len(dataset)
    for i in range(length):
        if debug:
            print(f'evaluating example {i} of {length}')
        qid, question, context = dataset['qid'][i], dataset['question'][i], dataset['context'][i]
        input_ids = tokenizer.encode('question: %s  context: %s' % (question, context), 
                                return_tensors='pt').to(model.device)
        decoder_input_ids = torch.tensor([[tokenizer.encode(tokenizer.pad_token)[0]]]).to(input_ids.device)
        for i in range(10):
            idx = model(input_ids, decoder_input_ids=decoder_input_ids, return_dict=True).logits.argmax(-1)[0][-1]
            decoder_input_ids=torch.cat((decoder_input_ids,torch.tensor([[idx]]).to(decoder_input_ids.device)), dim=1)
        pred = ' '.join([tokenizer.decode(decoder_input_ids[0], skip_special_tokens=False)])
        pred = pred.replace('</s>','').replace('<pad>','').lower().strip()
        predictions[qid] = pred
    return predictions

def compute_metrics(wrapper, debug=False):
    model, tokenizer, val_set, test_set = wrapper['model'], wrapper['tokenizer'], wrapper['val_set'], wrapper['test_set']

    val_set_gts = dict(zip(val_set['qid'], val_set['answers']))
    val_set_pred = generate_predictions(model, tokenizer, val_set, debug)
    val_set_metric = evaluate(val_set_gts, val_set_pred, True)
    print(f'     val_set: {val_set_metric}')
    
    test_set_gts = dict(zip(test_set['qid'], test_set['answers']))
    test_set_pred = generate_predictions(model, tokenizer, test_set, debug)
    test_set_metric = evaluate(test_set_gts, test_set_pred, True)
    print(f'     test_set: {test_set_metric}')
    
    return val_set_metric, test_set_metric

def save_logs(model, args):
    if args['logging']:
        model.save_soft_prompt(args['path'], filename='soft_prompt.model')
        metainfo_file = os.path.join(args['path'], 'info.json')
        with open(metainfo_file, 'w') as fp:
            json.dump(args, fp)

def test_gpt(wrapper, args):
    model, tokenizer = wrapper['model'], wrapper['tokenizer']
    val_path = './prefix_data/webnlg_challenge_2017/dev.json'
    test_path = './prefix_data/dart/dart-v1.1.1-full-test.json'
    val, test = read_webnlg_files(val_path, tokenizer), read_triples_files(test_path, tokenizer)
    for prompt_text_dict in [val, test]:
        prompt_text_pair = list(prompt_text_dict.keys())
        prompt_text_lst, prompt_rela_lst = zip(*prompt_text_pair)
        for prompt_idx, prompt_text in enumerate(prompt_text_lst):
            print(prompt_text)
            input_ids = tokenizer.encode(prompt_text, 
                                return_tensors='pt').to(model.device)
            length = input_ids.shape[-1]
            for i in range(10):
                output = model(input_ids)
                output = output.logits.argmax(-1)[0][-1].unsqueeze(-1).unsqueeze(-1)
                input_ids = torch.cat([input_ids, output], dim=-1)
            output_ids = input_ids[:,length:]
            print(tokenizer.batch_decode(output_ids))
            print()


def run(args):
    args['path'] = None
    path = setup_logs(args)
    model_data_wrapper = get_data_model(args)
    optim_wrapper = get_optim(model_data_wrapper['model'], args)
    training_args = get_training_args(args)
    if args['task'] == 'qa':
        data_collator = T2TDataCollator()
    if args['task'] == 't2t':
        data_collator = DataCollatorForData2TextLanguageModeling(
            tokenizer=model_data_wrapper["tokenizer"],
            mlm=False,
            mlm_probability=0.15,
            format_mode="cat",
        )
    trainer = Trainer(
                model=model_data_wrapper['model'],
                args=training_args,
                train_dataset=model_data_wrapper['train_set'],
                eval_dataset=model_data_wrapper['val_set'],
                data_collator=data_collator,
                optimizers=(optim_wrapper['optim'], optim_wrapper['scheduler']),
            )
    trainer.train()
    save_logs(model_data_wrapper['model'], args)
    test_gpt(model_data_wrapper, args)
    # TODO: add evaluation on validation set and the test set
    # compute_metrics(model_data_wrapper)

def test_model(args):
    # TODO: remove hardcoding directory
    # model = T5PromptTuningLM.from_pretrained(args["model"], 
    #                                                  return_dict=True,
    #                                                  soft_prompt_path='prompt_tuning/SQuAD/t5-small/1/2022-02-04-221142/soft_prompt.model')
    # tokenizer = T5Tokenizer.from_pretrained(args["model"])
    # train_set, val_set, test_set = get_dataset(tokenizer, args)
    # wrapper = {
    #     'model': model,
    #     'tokenizer': tokenizer,
    #     'train_set': train_set,
    #     'val_set': val_set,
    #     'test_set': test_set,
    # }
    # compute_metrics(wrapper, True)
    print(args)
    device = torch.device("cuda")
    tokenizer = GPT2Tokenizer.from_pretrained(args["model"])
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2PromptTuningLM.from_pretrained(
            args["model"],
            return_dict=True,
            soft_prompt_path='/datasets/home/37/137/ziw029/T5_SQuAD_Prompt_Tuning/prompt_tuning/webnlg/gpt2-medium/10/2022-03-03-214002/soft_prompt.model'
        )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    # tokenizer.to(device)
    wrapper = {
        'model': model,
        'tokenizer': tokenizer,
    }
    test_gpt(wrapper, args)




