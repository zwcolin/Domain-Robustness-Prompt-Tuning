import os, sys

# example: python train_run.py keyword temp_keyword _
if __name__ == '__main__':
    mode = sys.argv[1]
    control_mode = sys.argv[2]
    eval_split = sys.argv[3]
    model_file = None
    old_model = None
    MODEL_FILE = sys.argv[4]
    submit_job = (sys.argv[5] == 'yes')

    if mode == 'webnlg':
        Token_FILE = MODEL_FILE
        # gen_dir = 'webNLG_results'
        gen_dir = 'webNLG_results2'

        tuning_mode = 'prefixtune'
        app = '--optim_prefix {} --preseqlen {} '.format('yes', 20)
        app += "--prefix_mode activation "
        app += " --format_mode cat "

        MODEL_FILE2 = '/home/l6wang/T5_SQuAD_Prompt_Tuning/webnlg_models/train'

        # MODEL_FILE = 'gpt2-large'
        MODEL_FILE = 'gpt2-medium'


    COMMANDLINE = "python run_generation.py \
        --model_type=gpt2 \
        --length 100 \
        --model_name_or_path={} \
        --num_return_sequences 5 \
        --stop_token [EOS] \
        --tokenizer_name={} \
        --task_mode={} \
        --control_mode={} --tuning_mode {} --gen_dir {} --eval_dataset {} \
    ".format(MODEL_FILE, Token_FILE, mode, control_mode, tuning_mode, gen_dir, eval_split)

    COMMANDLINE += app
    COMMANDLINE += ' --prefixModel_name_or_path {}'.format(MODEL_FILE2)


    if MODEL_FILE == 'gpt2-large':
        COMMANDLINE += ' --cache_dir cache/gpt2-large-s3 '

    if MODEL_FILE == 'gpt2-medium':
        COMMANDLINE += ' --cache_dir cache/gpt2-medium-s3 '


    print(COMMANDLINE)

    if not submit_job:
        os.system(COMMANDLINE)

