# R3++ End-to-End Pipeline (Download → Build → Train → Eval)

This runbook assumes you want to train a **single unified model** across ScreenQA + ChartQA + InfoVQA.
It also flags current limitations around full fine-tuning on 8x V100.

## 0) Environment Setup

```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets pillow sentence-transformers peft
pip install huggingface-hub
pip install faiss-gpu  # or faiss-cpu if you do not have CUDA
```

## 0.5) Download Retrieval Backbones (CLIP + Text Encoder)

```bash
python scripts/download_retrieval_models.py --out_dir models
```

This writes:
- `models/clip-vit-b32-laion2B`
- `models/all-MiniLM-L6-v2`

These paths are now the default in `config/train_config.py`.

## 1) Download Raw Datasets

```bash
python scripts/download_screenqa.py --out_dir data/raw/screenqa
python scripts/download_chartqa.py --out_dir data/raw/chartqa
python scripts/download_infographicvqa.py --out_dir data/raw/infovqa
```

This produces:
- `data/raw/<dataset>/images/*.png`
- `data/raw/<dataset>/*_raw_<split>.jsonl`

## 2) Build Unified JSONL (with image_prefix)

## 2a) Build OCR Cache (Recommended for high-quality pseudo-text)

Install OCR engine (recommended):
```bash
pip install paddleocr
```

Optional (GPU): install paddlepaddle for CUDA that matches your system.

Build OCR cache per dataset (uses existing raw JSONL + images):
```bash
python scripts/build_ocr_cache.py \
  --raw_dir data/raw/screenqa \
  --image_root data/raw/screenqa \
  --out_path data/ocr/screenqa_ocr.jsonl \
  --engine paddleocr \
  --lang en \
  --use_angle_cls \
  --use_gpu

python scripts/build_ocr_cache.py \
  --raw_dir data/raw/chartqa \
  --image_root data/raw/chartqa \
  --out_path data/ocr/chartqa_ocr.jsonl \
  --engine paddleocr \
  --lang en \
  --use_angle_cls \
  --use_gpu

python scripts/build_ocr_cache.py \
  --raw_dir data/raw/infovqa \
  --image_root data/raw/infovqa \
  --out_path data/ocr/infovqa_ocr.jsonl \
  --engine paddleocr \
  --lang en \
  --use_angle_cls \
  --use_gpu
```

Then rebuild unified JSON with OCR:
```bash
python data/preprocess/build_unified_json.py \
  --dataset screenqa \
  --raw_dir data/raw/screenqa \
  --out_dir data/unified \
  --image_prefix screenqa \
  --ocr_cache data/ocr/screenqa_ocr.jsonl

python data/preprocess/build_unified_json.py \
  --dataset chartqa \
  --raw_dir data/raw/chartqa \
  --out_dir data/unified \
  --image_prefix chartqa \
  --ocr_cache data/ocr/chartqa_ocr.jsonl

python data/preprocess/build_unified_json.py \
  --dataset infovqa \
  --raw_dir data/raw/infovqa \
  --out_dir data/unified \
  --image_prefix infovqa \
  --ocr_cache data/ocr/infovqa_ocr.jsonl
```

Or use the one-shot helper (rebuild unified + indices):
```bash
python scripts/rebuild_unified_indices.py \
  --raw_root data/raw \
  --out_dir data/unified \
  --image_root data/raw \
  --index_dir indices \
  --ocr_root data/ocr
```

To **merge datasets into a single training set**, make sure each sample keeps a dataset prefix so
`image_root` can be shared across datasets.

```bash
python data/preprocess/build_unified_json.py \
  --dataset screenqa \
  --raw_dir data/raw/screenqa \
  --out_dir data/unified \
  --image_prefix screenqa

python data/preprocess/build_unified_json.py \
  --dataset chartqa \
  --raw_dir data/raw/chartqa \
  --out_dir data/unified \
  --image_prefix chartqa

python data/preprocess/build_unified_json.py \
  --dataset infovqa \
  --raw_dir data/raw/infovqa \
  --out_dir data/unified \
  --image_prefix infovqa
```

Resulting `image_path` will look like:
- `screenqa/images/train_0.png`
- `chartqa/images/validation_123.png`
- `infovqa/images/test_4.png`

Then set `image_root=data/raw` during training/evaluation.

## 3) Merge Unified JSONL Files

```bash
cat data/unified/screenqa_unified_train.jsonl \
    data/unified/chartqa_unified_train.jsonl \
    data/unified/infovqa_unified_train.jsonl \
  > data/unified/train.jsonl

cat data/unified/screenqa_unified_val.jsonl \
    data/unified/chartqa_unified_val.jsonl \
    data/unified/infovqa_unified_test.jsonl \
  > data/unified/val.jsonl
```

## 4) Build Retrieval Indices (FAISS)

```bash
python scripts/build_indices.py \
  --jsonl data/unified/train.jsonl \
  --image_root data/raw \
  --out_dir indices
```

## 4b) Build Retrieval Indices (Multi-GPU Sharding)

```bash
torchrun --nproc_per_node=8 scripts/build_indices.py \
  --jsonl data/unified/train.jsonl \
  --image_root data/raw \
  --out_dir indices \
  --num_shards 8 \
  --batch_size 32

python scripts/build_indices.py --out_dir indices --merge_shards 8
```

## 5) Train (Single GPU / LoRA recommended first)

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/train_r3.py \
  --model_name models/Qwen3-VL-8B-Instruct \
  --train_jsonl data/unified/train.jsonl \
  --val_jsonl data/unified/val.jsonl \
  --image_root data/raw \
  --index_dir indices \
  --top_k 3 \
  --fp16 \
  --batch_size 1 \
  --sampling_alpha 0.5 \
  --adam_eps 1e-6 \
  --use_lora \
  --lora_targets v_proj,o_proj \
  --disable_teacher \
  --gradient_checkpointing
```

## 5b) Train with FSDP (8x V100, LoRA recommended)

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=8 scripts/train_r3.py \
  --backend fsdp \
  --model_name models/Qwen3-VL-8B-Instruct \
  --train_jsonl data/unified/train.jsonl \
  --val_jsonl data/unified/val.jsonl \
  --image_root data/raw \
  --index_dir indices \
  --top_k 3 \
  --fp16 \
  --batch_size 1 \
  --sampling_alpha 0.5 \
  --grad_accum 4 \
  --fsdp_min_params 10000000 \
  --adam_eps 1e-6 \
  --use_lora \
  --lora_targets v_proj,o_proj \
  --disable_teacher \
  --gradient_checkpointing
```

## 5c) Train with DeepSpeed ZeRO-3 (8x V100)

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

deepspeed --num_gpus=8 scripts/train_r3.py \
  --backend deepspeed \
  --deepspeed_config configs/deepspeed_zero3.json \
  --model_name models/Qwen3-VL-8B-Instruct \
  --train_jsonl data/unified/train.jsonl \
  --val_jsonl data/unified/val.jsonl \
  --image_root data/raw \
  --index_dir indices \
  --top_k 3 \
  --fp16 \
  --batch_size 1 \
  --sampling_alpha 0.5 \
  --grad_accum 4 \
  --adam_eps 1e-6 \
  --use_lora \
  --lora_targets v_proj,o_proj \
  --disable_teacher \
  --gradient_checkpointing
```

Note: DeepSpeed checkpoints are sharded. To export a full FP16 model for evaluation, use:
```bash
python -m deepspeed.utils.zero_to_fp32 \
  --input_dir checkpoints/step_1000 \
  --output_file checkpoints/step_1000/pytorch_model.bin
```

## 6) Evaluate

```bash
python scripts/eval_r3.py \
  --checkpoint_dir checkpoints/step_1000 \
  --model_name models/Qwen3-VL-8B-Instruct \
  --val_jsonl data/unified/val.jsonl \
  --image_root data/raw \
  --index_dir indices \
  --top_k 3
```

## 6b) Evaluate Base vs R3 (Clean + Corrupted)

### Base model (clean, no pseudo-text)
```bash
python scripts/eval_r3.py \
  --eval_mode base \
  --clean_only \
  --no_pseudo_text \
  --model_name models/Qwen3-VL-8B-Instruct \
  --val_jsonl data/unified/val.jsonl \
  --image_root data/raw \
  --index_dir indices \
  --top_k 3
```

### Base model only (corruption sweep, no R3 / no retrieval)
```bash
python scripts/eval_r3.py \
  --eval_mode base \
  --corruption_levels 0,0.2,0.4,0.6,0.8 \
  --no_pseudo_text \
  --corrupt_text_target question \
  --model_name models/Qwen3-VL-8B-Instruct \
  --val_jsonl data/unified/val.jsonl \
  --image_root data/raw \
  --index_dir indices \
  --top_k 3 \
  --out_json results/base_corrupt.json
```

### Base model only (image-only corruption)
```bash
python scripts/eval_r3.py \
  --eval_mode base \
  --corruption_levels 0,0.2,0.4,0.6,0.8 \
  --no_pseudo_text \
  --corrupt_text_target none \
  --model_name models/Qwen3-VL-8B-Instruct \
  --val_jsonl data/unified/val.jsonl \
  --image_root data/raw \
  --index_dir indices \
  --top_k 3 \
  --out_json results/base_corrupt_image_only.json
```

### R3 model (clean)
```bash
python scripts/eval_r3.py \
  --eval_mode r3 \
  --clean_only \
  --checkpoint_dir checkpoints/step_1000 \
  --model_name models/Qwen3-VL-8B-Instruct \
  --val_jsonl data/unified/val.jsonl \
  --image_root data/raw \
  --index_dir indices \
  --top_k 3
```

### R3 model (corruption sweep)
```bash
python scripts/eval_r3.py \
  --eval_mode r3 \
  --corruption_levels 0,0.2,0.4,0.6,0.8 \
  --checkpoint_dir checkpoints/step_1000 \
  --model_name models/Qwen3-VL-8B-Instruct \
  --val_jsonl data/unified/val.jsonl \
  --image_root data/raw \
  --index_dir indices \
  --top_k 3
```

### R3 ablation example (disable retrieval + prefix + gate)
```bash
python scripts/eval_r3.py \
  --eval_mode r3 \
  --clean_only \
  --disable_text_retrieval \
  --disable_image_retrieval \
  --disable_prefix \
  --disable_gate \
  --checkpoint_dir checkpoints/step_1000 \
  --model_name models/Qwen3-VL-8B-Instruct \
  --val_jsonl data/unified/val.jsonl \
  --image_root data/raw \
  --index_dir indices \
  --top_k 3
```

## Multi-GPU / Model-Parallel Notes (8x V100)

- **Use fp16 on V100** (bf16 is not supported).
- **LoRA + gradient checkpointing + disable_teacher** are the fastest ways to avoid OOM.
- FSDP and DeepSpeed ZeRO-3 are integrated; pipeline/tensor parallel is not.
