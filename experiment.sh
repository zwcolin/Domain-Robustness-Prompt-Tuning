python -u run_prompt_tuning.py \
    --method prompt_tuning \
    --model t5-base \
    --n_tokens 50 \
    --mode train \
    --bz 7 \
    --epoch 0.03 \
    --logging 0
# more arguments can be added

# train prefixtuning with gpt2-medium on webnlg
# python run_prefix_tuning.py --logging 1

# test prefixtuning with gpt2-medium on bart
#  python run_prefix_tuning.py --mode test --test_set dart

# python run.py \
#     --method prompt_tuning \
#     --model t5-small \
#     --n_tokens 10 \
#     --mode train \
#     --dataset squad \

