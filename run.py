import sys

from prepare_data import create_or_load
from collator import T2TDataCollator
from transformers import AdamW, get_scheduler, Trainer, TrainingArguments, Adafactor
from transformers import T5Tokenizer
from model import T5PromptTuningLM

if __name__ == "__main__":

    model_name = sys.argv[1]
    n_tokens = int(sys.argv[2])
    batch_size = int(sys.argv[3])

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