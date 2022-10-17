#!/bin/bash

PORT_ID=$(expr $RANDOM + 2000)

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=2

python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/nli_for_simcse.csv \
    --output_dir result/diffaug-sup-bert \
    --num_train_epochs 3 \
    --per_device_train_batch_size 128 \
    --gradient_accumulation_steps 4 \
    --per_device_sup_train_batch_size 128 \
    --learning_rate 5e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 50 \
    --save_steps 50 \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    --sup_learning_rate 1e-3 \
    --seed  0 \
    --mlp_train \
    --max_seq_length  32 \
    --apply_prompt \
    --prompt_template_id  00 \
    --apply_template_delta_train \
    --use_two_datasets \
    --sup_data_sample_ratio  1 \
    --sup_label_num  2 \
    --phase1_steps  1250 \
    --use_prefix  \
    --prefix_len  16 \
    --use_two_optimizers \
    $@
