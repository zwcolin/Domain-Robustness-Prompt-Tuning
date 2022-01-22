import sys, os
from datetime import datetime
from contextlib import redirect_stdout
import json
import argparse

from prepare_data import create_or_load
from collator import T2TDataCollator
from transformers import get_scheduler, Trainer, TrainingArguments
from transformers.optimization import Adafactor
from transformers import T5Tokenizer, GPT2Tokenizer
from model import T5PromptTuningLM
import torch

parser = argparse.ArgumentParser()

# meta-information, or args that specific to all tuning methods
parser.add_argument('--method', default='prompt_tuning', type=str, 
                    help='Tuning method being used')
parser.add_argument('--model', default='t5-small', type=str,
                    help='Model being used')
parser.add_argument('--mode', default='train', type=str,
                    help='Mode being used')
parser.add_argument('--dataset', default='squad', type=str,
                    help='Dataset being used')
parser.add_argument('--model_dir', default='none', type=str,
                    help='Prompt or prefix being used for testing')

# specific-to-prompt-tuning
parser.add_argument('--soft_prompt_path', default=None, type=str,
                    help='the path of a tuned soft prompt')
parser.add_argument('--n_tokens', default=10, type=int,
                    help='number of tokens')
parser.add_argument('--initialize_from_vocab', default=True, type=bool,
                    help='if the initial prompt is initialized from existing vocabulary')
parser.add_argument('--random_range', default=0.5, type=float,
                    help='weight range from a uniform distribution if not initialized from existing vocabulary')

# specific-to-prefix-tuning
# TODO

# hyperparameters for fine-tuning
parser.add_argument('--bz', default=16, type=int,
                    help='batch size')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='learning rate')
parser.add_argument('--epoch', default=4, type=int,
                    help='number of epochs')
parser.add_argument('--optimizer', default='adafactor', type=str,
                    help='which optimizer to use')
parser.add_argument('--clip_threshold', default=1., type=float,
                    help='Threshold of root mean square of final gradient update')
parser.add_argument('--scale_parameter', default=False, type=bool,
                    help='If True, learning rate is scaled by root mean square')
parser.add_argument('--relative_step', default=False, type=bool,
                    help='If True, time-dependent learning rate is computed instead of external learning rate')
parser.add_argument('--warmup_init', default=False, type=bool,
                    help='Time-dependent learning rate computation depends on whether warm-up initialization is being used')

args = vars(parser.parse_args())

def train(args):
    # logging
    timestamp = datetime.now().strftime('%Y-%m-%d-%H%M%S')
    path = '{}/{}/{}/{}/{}'.format(args['method'], args['dataset'], args['model'], args['n_tokens'], timestamp)
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    sys.stderr = open(os.path.join(path, 'log.txt'), "w")
    sys.stdout = open(os.path.join(path, 'log.txt'), "w")

    if 't5' in args['model']:
        tokenizer = T5Tokenizer.from_pretrained(args['model'])
        train_dataset, valid_dataset = create_or_load(tokenizer)
        model = T5PromptTuningLM.from_pretrained(
            args['model'],
            n_tokens=args['n_tokens'],
            soft_prompt_path=args['soft_prompt_path'],
            initialize_from_vocab=args['initialize_from_vocab'],
            random_range=args['random_range'],)

    if 'gpt' in args['model']:
        tokenizer = GPT2Tokenizer.from_pretrained(args['model'])
        train_dataset, valid_dataset = create_or_load(tokenizer)
        model = None
        # TODO: add GPT model here

    if args['optimizer'] == 'adafactor':
        optimizer = Adafactor(model.parameters(), 
            scale_parameter=args['scale_parameter'], 
            relative_step=args['relative_step'], 
            warmup_init=args['warmup_init'], 
            lr=args['lr'],
            clip_threshold=args['clip_threshold'])
        # TODO: we might need s scheduler
        lr_scheduler = None

    elif args['optimizer'] == 'adamw':
        pass 
        # TODO: get adamw optimizer for gpt2
    else:
        pass

    training_args = TrainingArguments(
            per_device_train_batch_size=args['bz'],  # batch size per device during training
            per_device_eval_batch_size=args['bz'],   # batch size for evaluation
            num_train_epochs=args['epoch'],
            # num_train_epochs=args['epoch'],
            disable_tqdm=True,

            output_dir='./results',          # output directory
            save_steps=1000000, #TODO: hardcoded for debugging, I don't want to mess up my disk space

            logging_dir=path,            # directory for storing logs
            logging_steps=20,
        )
    
    trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                data_collator=T2TDataCollator(),
                optimizers=(optimizer, lr_scheduler),
            )

    trainer.train()

    if args['method'] == 'prompt_tuning':
        model.save_soft_prompt(path, filename='soft_prompt.model')
        metainfo_file = os.path.join(path, 'info.json')
        with open(metainfo_file, 'w') as fp:
            json.dump(args, fp)
        sys.stdout.close()
    else:
        pass # TODO: do the same for prefix tuning

def main(args):
    if args['mode'] == 'train':
        train(args)
    else:
        if 'model' in targets:
            try:
                model_name = str(targets[1])
                n_tokens = int(targets[2])
                model = T5PromptTuningLM.from_pretrained(model_name, 
                                                        return_dict=False,
                                                        soft_prompt_path=f'soft_prompt/soft_prompt_{model_name}_{n_tokens}.model')
            except:
                print('Please read the README.md to learn how to run the script properly!!')
                model = T5PromptTuningLM.from_pretrained('t5-small', 
                                                    return_dict=False,
                                                    soft_prompt_path='soft_prompt/soft_prompt_t5-small_10.model')
                print('Specified configuration failed to load... Load default settings: model_name=t5-small, n_tokens=10')
            
        if 'test' in targets:
            try:
                model_name = str(targets[1])
                n_tokens = int(targets[2])
                model = T5PromptTuningLM.from_pretrained(model_name, 
                                                        return_dict=False,
                                                        soft_prompt_path=f'soft_prompt/soft_prompt_{model_name}_{n_tokens}.model')
                tokenizer = T5Tokenizer.from_pretrained(model_name)
            except:
                print('Please read the README.md to learn how to run the script properly!!')
                model = T5PromptTuningLM.from_pretrained('t5-small', 
                                                    return_dict=False,
                                                    soft_prompt_path='soft_prompt/soft_prompt_t5-small_10.model')
                tokenizer = T5Tokenizer.from_pretrained('t5-small')
                print('Specified configuration failed to load... Load default settings: model_name=t5-small, n_tokens=10')

            model = T5PromptTuningLM.from_pretrained('t5-small', 
                                                    return_dict=False,
                                                    soft_prompt_path='soft_prompt/soft_prompt_t5-small_10.model')
            tokenizer = T5Tokenizer.from_pretrained('t5-small')
            train_dataset, valid_dataset = create_or_load(tokenizer)
            for i in range(10):
                print('------------------------------------')
                question, context = valid_dataset['question'][i], valid_dataset['context'][i]
                input_ids = tokenizer.encode('question: %s  context: %s' % (question, context), 
                                        return_tensors='pt').to(model.device)
                answers = valid_dataset['answers'][i]['text']
                for i in range(len(answers)):
                    answers[i] = answers[i].lower().strip()
                print(f'context: {context}')
                print()
                print(f'question: {question}')
                print()
                print(f'answers: {answers}')
                decoder_input_ids = torch.tensor([[tokenizer.encode(tokenizer.pad_token)[0]]]).to(input_ids.device)
                for i in range(10):
                    idx = model(input_ids, decoder_input_ids=decoder_input_ids, return_dict=True).logits.argmax(-1)[0][-1]
                    decoder_input_ids=torch.cat((decoder_input_ids,torch.tensor([[idx]]).to(decoder_input_ids.device)), dim=1)
                pred = ' '.join([tokenizer.decode(decoder_input_ids[0], skip_special_tokens=False)])
                pred = pred.replace('</s>','').replace('<pad>','')

                print(f'model prediction: {pred.lower().strip()}')
    
if __name__ == '__main__':
    main(args)
