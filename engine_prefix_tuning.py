import sys, os
from datetime import datetime
import torch
from prepare_data import get_dataset as get_dataset_qa
from prepare_webnlg import get_dataset as get_dataset_t2t
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    AutoConfig,
    AutoTokenizer,
    DataCollatorForData2TextLanguageModeling
)
from transformers.optimization import Adafactor
from model_prefix_tuning import PrefixTuning, T5PrefixTuning
from trainer_prefix import Trainer_Prefix
from collator import T2TDataCollator
from utils_prompt_tuning import evaluate


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
        model_name = args["model"]
        config = AutoConfig.from_pretrained(
            model_name, cache_dir=f"cache/{model_name}-s3"
        )
        config.model = model_name
        config.preseqlen = args["preseqlen"]
        model = T5PrefixTuning(config)

        if args['task'] == 'qa':
            train_set, val_set, test_set = get_dataset_qa(model.tokenizer, args)

        return {
            'model': model,
            'tokenizer': model.tokenizer,
            'train_set': train_set,
            'val_set': val_set,
            'test_set': test_set,
        }

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
        train_dataset = get_dataset_t2t(
            tokenizer=tokenizer,
            file_path="./prefix_data/webnlg_challenge_2017/train.json",
        )
        eval_dataset = get_dataset_t2t(
            tokenizer=tokenizer,
            file_path="./prefix_data/webnlg_challenge_2017/dev.json",
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
    if 'gpt2' in args["model"]:
        training_args = TrainingArguments(
            output_dir=f"./webnlg_models/{args['model']}/{args['preseqlen']}",
            overwrite_output_dir=False,
            do_train=True,
            do_eval=True,
            evaluate_during_training=True,
            evaluation_strategy="steps",
            prediction_loss_only=True,
            per_device_train_batch_size=args["bz"],
            per_device_eval_batch_size=args["bz"],
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
    else:
        # T5 training arguments
        training_args = TrainingArguments(
            output_dir=f"./qa_models/{args['model']}/{args['preseqlen']}",
            overwrite_output_dir=False,
            do_train=True,
            do_eval=True,
            evaluate_during_training=True,
            evaluation_strategy="steps",
            prediction_loss_only=True,
            per_device_train_batch_size=args["bz"],
            per_device_eval_batch_size=args["bz"],
            num_train_epochs=args["epoch"],
            logging_dir="./qa_models/runs/",
            logging_steps=100,
            save_steps=500000,
            save_total_limit=1,
            seed=101,
            eval_steps=5000,
            dataloader_num_workers=0,
            disable_tqdm=True,
            remove_unused_columns=False,
        )

    return training_args

def generate_predictions(model, tokenizer, dataset, debug):
    predictions = {}
    length = 10 if debug else len(dataset)
    for i in range(length):
        if debug:
            print(f'evaluating example {i} of {length}')
        qid, question, context = dataset['qid'][i], dataset['question'][i], dataset['context'][i]
        input_ids = tokenizer.encode('question: %s  context: %s' % (question, context), max_length=512,
                                     truncation=True, return_tensors='pt').to(model.device) 
        decoder_input_ids = torch.tensor([[tokenizer.encode(tokenizer.pad_token)[0]]]).to(input_ids.device)
        output = model.generate(input_ids,  decoder_input_ids=decoder_input_ids, return_dict=True).to(input_ids.device)
        pred = ' '.join([tokenizer.decode(output[0], skip_special_tokens=False)])
        pred = pred.replace('</s>','').replace('<pad>','').lower().strip()
        predictions[qid] = pred
    return predictions

def compute_metrics(wrapper, debug=False):
    model, tokenizer, val_set, test_set = wrapper['model'], wrapper['tokenizer'], wrapper['val_set'], wrapper['test_set']
    model.cuda()
    val_set_gts = dict(zip(val_set['qid'], val_set['answers']))
    val_set_pred = generate_predictions(model, tokenizer, val_set, debug)
    val_set_metric = evaluate(val_set_gts, val_set_pred, True)
    print(f'     val_set: {val_set_metric}')
    
    test_set_gts = dict(zip(test_set['qid'], test_set['answers']))
    test_set_pred = generate_predictions(model, tokenizer, test_set, debug)
    test_set_metric = evaluate(test_set_gts, test_set_pred, True)
    print(f'     test_set: {test_set_metric}')
    return val_set_metric, test_set_metric


def run(args):
    if os.path.exists('./prefix_data'):
        pass
    else:
        os.system('git clone https://github.com/wanglec/prefix_data.git') 
    args["path"] = None
    path = setup_logs(args)
    training_args = get_training_args(args)


    if 'gpt2' in args["model"]:
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

        elif args["mode"] == "test":
            checkpoint_path = os.path.abspath(training_args.output_dir)
            print("running evaluation on ", checkpoint_path)

            if args["test_set"] == "webnlg":
                # print("python gen.py webnlg yes valid {} no".format(checkpoint_path))
                print("python gen.py webnlg yes test {} no".format(checkpoint_path))
                # os.system("python gen.py webnlg yes valid {} no".format(checkpoint_path))
                os.system("python gen.py webnlg yes test {} no".format(checkpoint_path))

            elif args["test_set"] == "dart":
                print("python gen.py triples yes test {} no".format(checkpoint_path))
                os.system("python gen.py triples yes test {} no".format(checkpoint_path))


    elif 't5' in args["model"]:
        if args["mode"] == "train": 
            model_data_wrapper = get_data_model(args)
            model=model_data_wrapper["model"]
            tokenizer = model_data_wrapper["tokenizer"]
            train_dataset=model_data_wrapper["train_set"]
            val_dataset=model_data_wrapper["val_set"]
            test_dataset=model_data_wrapper["test_set"]

            optimizer = Adafactor(
                model.parameters(),
                scale_parameter=False, 
                relative_step=False, 
                warmup_init=False, 
                lr=1e-4,
                clip_threshold=1.0)
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                prediction_loss_only=True,
                data_collator=T2TDataCollator(),
                optimizers=(optimizer, None),
            )
            if trainer.is_world_master():
                tokenizer.save_pretrained(training_args.output_dir)
            trainer.train()
            trainer.save_model()
        
        elif args["mode"] == "test":
            model_name = args['model']
            config = AutoConfig.from_pretrained(
                model_name, cache_dir=f"./cache/{model_name}-s3"
            )
            config.model = model_name
            config.preseqlen = args['preseqlen']
            model = T5PrefixTuning.from_pretrained(f"./qa_models/{model_name}/{config.preseqlen}", config=config)
            train_set, val_set, test_set = get_dataset_qa(model.tokenizer, args)
            wrapper = {
                'model': model,
                'tokenizer': model.tokenizer,
                'train_set': train_set,
                'val_set': val_set,
                'test_set': test_set,
            }
            compute_metrics(wrapper, True)



    