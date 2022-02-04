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
)
from model_prompt_tuning import T5PromptTuningLM, GPT2PromptTuningLM
from utils_prompt_tuning import evaluate
from prepare_data import get_dataset
from collator import T2TDataCollator

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
        model = GPT2PromptTuningLM.from_pretrained(
            args["model"],
            n_tokens=args["n_tokens"],
            soft_prompt_path=args["soft_prompt_path"],
            initialize_from_vocab=args["initialize_from_vocab"],
            random_range=args["random_range"],
        )
    if args['task'] == 'qa':
        train_set, val_set, test_set = get_dataset(tokenizer, args)

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
    training_args = TrainingArguments(
            remove_unused_columns=False,
            per_device_train_batch_size=args['bz'],  # batch size per device during training
            per_device_eval_batch_size=args['bz'],   # batch size for evaluation
            num_train_epochs=args['epoch'],
            disable_tqdm=True,
            report_to=None,
            output_dir='./results',          # output directory
            save_steps=1000000, #TODO: hardcoded for debugging, I don't want to mess up my disk space

            logging_dir=args['path'],            # directory for storing logs
            logging_steps=20,
            
        )
    return training_args

def generate_predictions(model, tokenizer, dataset):
    predictions = {}
    for i in range(len(dataset)):
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

def compute_metrics(wrapper):
    model, tokenizer, val_set, test_set = wrapper['model'], wrapper['tokenizer'], wrapper['val_set'], wrapper['test_set']

    val_set_gts = dict(zip(val_set['qid'], val_set['answers']))
    val_set_pred = generate_predictions(model, tokenizer, val_set)
    val_set_metric = evaluate(val_set_gts, val_set_pred, True)
    print(f'val_set: {val_set_metric}')
    
    test_set_gts = dict(zip(test_set['qid'], test_set['answers']))
    test_set_pred = generate_predictions(model, tokenizer, test_set)
    test_set_metric = evaluate(test_set_gts, test_set_pred, True)
    print(f'test_set: {test_set_metric}')
    
    return val_set_metric, test_set_metric

def save_logs(model, args):
    if args['logging']:
        model.save_soft_prompt(args['path'], filename='soft_prompt.model')
        metainfo_file = os.path.join(args['path'], 'info.json')
        with open(metainfo_file, 'w') as fp:
            json.dump(args, fp)
        sys.stdout.close()

def run(args):
    args['path'] = None
    path = setup_logs(args)
    model_data_wrapper = get_data_model(args)
    optim_wrapper = get_optim(model_data_wrapper['model'], args)
    training_args = get_training_args(args)

    trainer = Trainer(
                model=model_data_wrapper['model'],
                args=training_args,
                train_dataset=model_data_wrapper['train_set'],
                eval_dataset=model_data_wrapper['val_set'],
                data_collator=T2TDataCollator(),
                optimizers=(optim_wrapper['optim'], optim_wrapper['scheduler']),
            )

    trainer.train()
    # TODO: add evaluation on validation set and the test set
    compute_metrics(model_data_wrapper)

    save_logs(model_data_wrapper['model'], args)
