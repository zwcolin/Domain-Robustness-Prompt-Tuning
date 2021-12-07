import sys

from prepare_data import create_or_load
from collator import T2TDataCollator
from transformers import AdamW, get_scheduler, Trainer, TrainingArguments, Adafactor
from transformers import T5Tokenizer
from model import T5PromptTuningLM
import torch

def main(targets):
    
    if 'train' in targets:
        model_name = str(targets[1])
        n_tokens = int(targets[2])
        batch_size = int(targets[3])

        tokenizer = T5Tokenizer.from_pretrained(model_name)
        train_dataset, valid_dataset = create_or_load(tokenizer)

        # if you want to train
        class Config:
            # Prompt-tuning
            n_prompt_tokens = n_tokens
            init_from_vocab = True
            # random_range = 0.5
        args = Config()

        model = T5PromptTuningLM.from_pretrained(
            model_name,
            n_tokens=args.n_prompt_tokens,
            initialize_from_vocab=args.init_from_vocab)

        # Set up training arguments, optimizers, etc
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if n == "soft_prompt.weight"],
                "lr": 1e-3,
                "scale_parameter": False,
                "relative_step": False,
            }
        ]
        optimizer = Adafactor(optimizer_grouped_parameters)
        lr_scheduler = get_scheduler(
            name='cosine',
            num_warmup_steps=0,
            optimizer=optimizer,
            num_training_steps=3,
        )

        training_args = TrainingArguments(
            output_dir='./results',          # output directory
            per_device_train_batch_size=batch_size,  # batch size per device during training
            per_device_eval_batch_size=batch_size*2,   # batch size for evaluation
            logging_dir='./logs',            # directory for storing logs
            logging_steps=100,
            save_steps=10000,
            report_to='tensorboard',
            prediction_loss_only=True,
            num_train_epochs=3,
        )

        # Initialize trainer

        trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                data_collator=T2TDataCollator(),
                optimizers=(optimizer, lr_scheduler),
            )

        # start training

        trainer.train()

        model.save_soft_prompt('soft_prompt', filename=f'soft_prompt_{model_name}_{n_tokens}.model')

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
    # run via:
    # python main.py data features model
    targets = sys.argv[1:]
    main(targets)
