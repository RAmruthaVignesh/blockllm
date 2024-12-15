HF_HOME=/workspace/.cache python -m torch.distributed.run \
    --standalone \
    --nproc_per_node 1 \
    torchrun_greedy.py \
    --model_config configs/llama_60m.json \
    --lr 1e-3 \
    --rank 256 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 60000 \
    --eval_every 500 \
    --optimizer blockllm \
    --warmup_steps 0
