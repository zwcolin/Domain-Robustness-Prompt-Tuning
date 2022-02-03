nohup python run.py \
    --method prompt_tuning \
    --model t5-base \
    --n_tokens 20 \
    --mode train \
    --bz 128 \
    --dataset squad &\
# more arguments can be added

# python run.py \
#     --method prompt_tuning \
#     --model t5-small \
#     --n_tokens 10 \
#     --mode train \
#     --dataset squad \
