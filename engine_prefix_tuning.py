import sys, os
from datetime import datetime
from prepare_webnlg import *
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    AutoConfig,
    AutoTokenizer,
    DataCollatorForData2TextLanguageModeling,
)
from model_prefix_tuning import PrefixTuning
from trainer_prefix import Trainer_Prefix


def setup_logs(args):
    if args["logging"]:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        path = "{}/{}/{}/{}/{}".format(
            args["method"],
            args["train_set"],
            args["model"],
            args["preseqlen"],
            timestamp,
        )
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        sys.stderr = open(os.path.join(path, "log.txt"), "w")
        sys.stdout = open(os.path.join(path, "log.txt"), "w")
        args["path"] = path
        return path


def get_data_model(args):
    if "t5" in args["model"]:
        pass
    if "gpt" in args["model"]:
        model_name = args["model"]
        config = AutoConfig.from_pretrained(
            model_name, cache_dir=f"cache/{model_name}-s3"
        )
        config._my_arg_tune_mode = "prefixtune"
        config._objective_mode = 1
        config._my_arg_task_mode = "webnlg"
        config.return_dict = True
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=f"cache/{model_name}-s3"
        )
        model = GPT2LMHeadModel.from_pretrained(
            model_name, config=config, cache_dir=f"cache/{model_name}-s3"
        )
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
        train_dataset = get_dataset(
            tokenizer=tokenizer,
            file_path="./data/webnlg_challenge_2017/train.json",
        )
        eval_dataset = get_dataset(
            tokenizer=tokenizer,
            file_path="./data/webnlg_challenge_2017/train.json",
        )
        for param in model.base_model.parameters():
            param.requires_grad = False
        gpt2 = model

        config_prefix = AutoConfig.from_pretrained(
            model_name, cache_dir=f"cache/{model_name}-s3"
        )
        config_prefix._my_arg_task_mode = "webnlg"
        config_prefix._my_arg_control = True
        config_prefix.train_weights = "no"
        config_prefix.optim_prefix = True
        config_prefix.preseqlen = args["preseqlen"]
        config_prefix.vocab_size = len(tokenizer)

        model = PrefixTuning(config_prefix, model_gpt2=gpt2)

    return {
        "model": model,
        "gpt2": gpt2,
        "tokenizer": tokenizer,
        "train_set": train_dataset,
        "eval_set": eval_dataset,
    }


def get_training_args(args):
    training_args = TrainingArguments(
        output_dir=f"./webnlg_models/{args['model']}/{args['preseqlen']}",
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        evaluate_during_training=True,
        evaluation_strategy="steps",
        prediction_loss_only=True,
        per_device_train_batch_size=args["epoch"],
        per_device_eval_batch_size=args["epoch"],
        adam_beta1=0.9,
        adam_beta2=0.999,
        num_train_epochs=args["epoch"],
        logging_dir="./webnlg_models/runs/",
        logging_steps=100,
        save_steps=500000,
        save_total_limit=1,
        seed=101,
        eval_steps=5000,
        dataloader_num_workers=0,
        run_name=None,
        disable_tqdm=True,
        remove_unused_columns=True,
        label_names=None,
    )
    return training_args


def run(args):
    args["path"] = None
    path = setup_logs(args)
    training_args = get_training_args(args)
    if args["mode"] == "train":
        model_data_wrapper = get_data_model(args)
        tokenizer = model_data_wrapper["tokenizer"]
        data_collator = DataCollatorForData2TextLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            mlm_probability=0.15,
            format_mode="cat",
        )
        trainer = Trainer_Prefix(
            model=model_data_wrapper["model"],
            tokenizer=tokenizer,
            model_gpt2=model_data_wrapper["gpt2"],
            args=training_args,
            prediction_loss_only=True,
            train_dataset=model_data_wrapper["train_set"],
            eval_dataset=model_data_wrapper["eval_set"],
            data_collator=data_collator,
            task_mode="webnlg",
            use_dropout=False,
        )

        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

        trainer.train()
        trainer.save_model()

    if args["mode"] == "test":
        checkpoint_path = os.path.abspath(training_args.output_dir)
        print("running evaluation on ", checkpoint_path)

        if args["test_set"] == "webnlg":
            print("python gen.py webnlg yes valid {} no".format(checkpoint_path))
            # print("python gen.py webnlg yes test {} no".format(checkpoint_path))
            os.system("python gen.py webnlg yes valid {} no".format(checkpoint_path))
            # os.system("python gen.py webnlg yes test {} no".format(checkpoint_path))

        elif args["test_set"] == "dart":
            print("python gen.py triples yes test {} no".format(checkpoint_path))
            os.system("python gen.py triples yes test {} no".format(checkpoint_path))
