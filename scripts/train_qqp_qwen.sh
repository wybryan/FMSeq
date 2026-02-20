

# # Qwen3-4B  (hidden 2560 → projected to hidden_dim)
# --vocab qwen  --use_plm_init qwen  --config_name Qwen/Qwen3-4B

# # Qwen3-8B  (hidden 4096 → projected to hidden_dim)
# --vocab qwen  --use_plm_init qwen  --config_name Qwen/Qwen3-8B

# # Qwen3-32B (hidden 5120 → projected to hidden_dim)
# --vocab qwen  --use_plm_init qwen  --config_name Qwen/Qwen3-32B

# # local checkpoint works the same way
# --vocab qwen  --use_plm_init qwen  --config_name /path/to/Qwen3-8B

python -m torch.distributed.launch --nproc_per_node=1 --master_port=12331 --use_env run_train.py \
  --diff_steps 2000 \
  --lr 0.0001 \
  --learning_steps 50000 \
  --save_interval 2000 \
  --seed 6868 \
  --noise_schedule sqrt \
  --hidden_dim 128 \
  --bsz 1024 \
  --microbatch 96 \
  --dataset qqp \
  --data_dir ./datasets/QQP-Official \
  --vocab        qwen \
  --use_plm_init qwen \
  --config_name  Qwen/Qwen3-32B \
  --seq_len 128 \
  --schedule_sampler lossaware \
  --resume_checkpoint "None" \
  --notes test-qqp-qwen \
  --sc_rate 0.5 \
  --rescale_max 1000