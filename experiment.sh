python run_porompt_tuning.py \
    --method prompt_tuning \
    --model t5-small \
    --n_tokens 1 \
    --mode train \
    --bz 20 \
    --epoch 0.001 \
    --logging 1
# more arguments can be added

# python run.py \
#     --method prompt_tuning \
#     --model t5-small \
#     --n_tokens 10 \
#     --mode train \
#     --dataset squad \

