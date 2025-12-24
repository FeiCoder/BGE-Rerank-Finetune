#!/bin/bash

# Note: The user provided example was for LLM-based reranker (FlagEmbedding.llm_reranker).
# BAAI/bge-reranker-v2-m3 is an encoder-only model (XLM-RoBERTa based), so we use the 
# FlagEmbedding.finetune.reranker.encoder_only.base module.
# We also removed LoRA arguments as they are typically not used/supported for this script 
# and full fine-tuning is feasible for this model size.

torchrun --nproc_per_node 1 --master_port 29501 \
-m FlagEmbedding.finetune.reranker.encoder_only.base \
--model_name_or_path /data/zf/models/BAAI/bge-reranker-v2-m3 \
--train_data ./data/toy_finetune_data.jsonl \
--output_dir ./output_model \
--learning_rate 2e-5 \
--num_train_epochs 1 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 1 \
--dataloader_drop_last True \
--query_max_len 128 \
--passage_max_len 128 \
--train_group_size 16 \
--logging_steps 1 \
--save_steps 2000 \
--save_total_limit 50 \
--ddp_find_unused_parameters False \
--gradient_checkpointing \
--deepspeed stage1.json \
--warmup_ratio 0.1 \
--bf16
