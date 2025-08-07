## Guidance for The Cold-Start Stage

### Dependencies

Please follow [readme](README_LF.md) file of LLaMA-Factory to handle the code environments.

### Prepare data

1. move data files (cold-start-zero-2k.json, cold-start-2k.json) to directory `./data/`.

2. register data as follows:

    ```
    "cold-start": {
      "file_name": "cold-start-2k.json",
      "formatting": "sharegpt",
      "columns": {
        "messages": "messages"
      },
      "tags": {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant",
        "system_tag": "system"
      }
    }
    ```

### Training Command

The learning rate for 1.7B and 32B model is 1e-5 and 5e-6 respectively. 

```shell
hostfile="./hostfile"
MODEL_PATH=<path_to_Model>

deepspeed --hostfile=$hostfile src/train.py \
    --stage sft \
    --do_train \
    --model_name_or_path $MODEL_PATH  \
    --dataset cold-start \
    --template qwen3 \
    --finetuning_type full \
    --output_dir <output_model_path> \
    --cache_dir .cache \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --cutoff_len 16000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.01 \
    --logging_steps 1 \
    --save_strategy epoch \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --seed 42 \
    --plot_loss \
    --flash_attn fa2 \
    --report_to tensorboard \
    --save_only_model True \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --bf16 True \
    --enable_entropy_sft \
    --token_ratio 0.8 \
    --entropy_coef 0.8 \
    --ddp_timeout 180000000
```
