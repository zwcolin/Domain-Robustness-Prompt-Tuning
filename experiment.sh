# CUDA_VISIBLE_DEVICES=5 nohup python -u run_prompt_tuning.py \
#     --method prompt_tuning \
#     --model t5-small \
#     --n_tokens 50 \
#     --mode train \
#     --bz 8 \
#     --epoch 4 \
#     --logging 1 &
# more arguments can be added

CUDA_VISIBLE_DEVICES=5 python -u run_prompt_tuning.py \
    --method prompt_tuning \
    --model t5-3b \
    --n_tokens 50 \
    --mode train \
    --bz 1 \
    --epoch 4 \
    --logging 0

# 1080
# small 24
# base 8
