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

# 1080
# small 24
# base 8

# CUDA_VISIBLE_DEVICES=4 nohup python -u run_prompt_tuning.py \
#         --method prompt_tuning \
#         --mode test \
#         --model t5-large \
#         --model_dir /datasets/home/37/137/ziw029/T5_SQuAD_Prompt_Tuning/prompt_tuning/SQuAD/t5-large/50/2022-02-05-131247 \
#         --logging 1 & \

# CUDA_VISIBLE_DEVICES=0 nohup python -u run_prompt_tuning.py \
#     --method prompt_tuning \
#     --model t5-base \
#     --n_tokens 10 \
#     --mode train \
#     --bz 4 \
#     --epoch 4 \
#     --logging 1 \
#     --train_set SQuAD \
#     --val_set SQuAD \
#     -tss NewsQA TriviaQA-web SearchQA HotpotQA BioASQ DROP RelationExtraction DuoRC.ParaphraseRC &

    # CUDA_VISIBLE_DEVICES=1 nohup python -u run_prompt_tuning.py \
    # --method prompt_tuning \
    # --model t5-base \
    # --n_tokens 10 \
    # --mode train \
    # --bz 4 \
    # --epoch 4 \
    # --logging 1 \
    # --train_set NewsQA \
    # --val_set NewsQA \
    # -tss SQuAD TriviaQA-web SearchQA HotpotQA BioASQ DROP RelationExtraction DuoRC.ParaphraseRC &

    # CUDA_VISIBLE_DEVICES=2 nohup python -u run_prompt_tuning.py \
    # --method prompt_tuning \
    # --model t5-base \
    # --n_tokens 10 \
    # --mode train \
    # --bz 4 \
    # --epoch 4 \
    # --logging 1 \
    # --train_set TriviaQA-web \
    # --val_set TriviaQA-web \
    # -tss SQuAD NewsQA SearchQA HotpotQA BioASQ DROP RelationExtraction DuoRC.ParaphraseRC &

    # CUDA_VISIBLE_DEVICES=3 nohup python -u run_prompt_tuning.py \
    # --method prompt_tuning \
    # --model t5-base \
    # --n_tokens 10 \
    # --mode train \
    # --bz 4 \
    # --epoch 4 \
    # --logging 1 \
    # --train_set SearchQA\
    # --val_set SearchQA\
    # -tss SQuAD TriviaQA-web NewsQA HotpotQA BioASQ DROP RelationExtraction DuoRC.ParaphraseRC &

    CUDA_VISIBLE_DEVICES=4 nohup python -u run_prompt_tuning.py \
    --method prompt_tuning \
    --model t5-base \
    --n_tokens 10 \
    --mode train \
    --bz 4 \
    --epoch 4 \
    --logging 1 \
    --train_set HotpotQA \
    --val_set HotpotQA \
    -tss SQuAD TriviaQA-web SearchQA NewsQA BioASQ DROP RelationExtraction DuoRC.ParaphraseRC &