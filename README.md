## Setting

### 1. Conda Env
```bash
conda create -n keye_captioning python=3.10
conda activate keye_captioning
```

### 3. Torch (CUDA 11.8)
```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118
```

> 🔗 PyTorch previous version download guide: https://pytorch.org/get-started/previous-versions/

### 4. Verification
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.version.cuda)"
```

### 5. Other Required Packages
```bash
pip install -r requirements.txt
```

### 6. Vision Processing Package
```bash
pip install --upgrade keye-vl-utils==1.5.2 -i https://pypi.org/simple
```

## Drive Mini Sample
[Download Drive Mini Sample](https://drive.google.com/drive/folders/1ZZfkhpWVY-U36Y5e62geOWX-euE2JpJx?usp=drive_link)

Move downloaded data to original folder data/

## Run Captioning
```bash
bash caption.sh
```

## Bash details
```bash
#!/bin/bash

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7} 

echo "Running captioning with CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Example usage of captioning.py
# Make sure to adjust paths as needed
python captioning.py \
    --video_paths_json "video_eval_paths/L1_video_paths.json" \
    --prompts_json "prompts.json" \
    --fps 1.0 \
    --max_frames 768 \
    --output_file "captions.jsonl"
```
```