import json, os, sys
from datetime import datetime
from transformers.optimization import Adafactor
from transformers import (
    T5Tokenizer,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
)
from model_prompt_tuning import T5PromptTuningLM, GPT2PromptTuningLM
from prepare_data import get_dataset
from collator import T2TDataCollator

def setup_logs(args):
    if args['logging']:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        path = "{}/{}/{}/{}/{}".format(args["method"], args["train_set"], args["model"], args["n_tokens"], timestamp)
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        sys.stderr = open(os.path.join(path, "log.txt"), "w")
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

            output_dir='./results',          # output directory
            save_steps=1000000, #TODO: hardcoded for debugging, I don't want to mess up my disk space

            logging_dir=args['path'],            # directory for storing logs
            logging_steps=20,
            
        )
    return training_args

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

    save_logs(model_data_wrapper['model'], args)
