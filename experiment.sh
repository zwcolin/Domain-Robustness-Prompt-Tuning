# CUDA_VISIBLE_DEVICES=5 nohup python -u run_prompt_tuning.py \
#     --method prompt_tuning \
#     --model t5-small \
#     --n_tokens 50 \
#     --mode train \
#     --bz 8 \
#     --epoch 4 \
#     --logging 1 &
# more arguments can be added

# CUDA_VISIBLE_DEVICES=5 python -u run_prompt_tuning.py \
#     --method prompt_tuning \
#     --model t5-3b \
#     --n_tokens 50 \
#     --mode train \
#     --bz 1 \
#     --epoch 4 \
#     --logging 0

# more arguments can be added

# train prefixtuning with gpt2-medium on webnlg
# python run_prefix_tuning.py --logging 1

# test prefixtuning with gpt2-medium on bart
#  python run_prefix_tuning.py --mode test --test_set dart
export PYTHONPATH=/datasets/home/37/137/ziw029/T5_SQuAD_Prompt_Tuning
python run_prompt_tuning.py \
    --method prompt_tuning \
    --model gpt2-medium \
    --task t2t \
    --n_tokens 1 \
    --mode train \
    --train_set webnlg \

# 1080
# small 24
# base 8
