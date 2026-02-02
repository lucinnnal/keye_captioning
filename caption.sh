#!/bin/bash

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5, 6} 

echo "Running captioning with CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Example usage of captioning.py
# Make sure to adjust paths as needed
python captioning.py \
    --video_paths_json "video_eval_paths/L1_video_paths.json" \
    --prompts_json "prompts.json" \
    --fps 1.0 \
    --max_frames 1024 \
    --output_file "captions.jsonl"
