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
- `models/clip-vit-base-patch32`
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

## 5) Train (Single GPU / LoRA recommended first)

```bash
python scripts/train_r3.py \
  --model_name models/Qwen3-VL-8B-Instruct \
  --train_jsonl data/unified/train.jsonl \
  --val_jsonl data/unified/val.jsonl \
  --image_root data/raw \
  --index_dir indices \
  --bf16 \
  --batch_size 1 \
  --use_lora
```

## 5b) Train with FSDP (8x V100, full fine-tune)

```bash
torchrun --nproc_per_node=8 scripts/train_r3.py \
  --backend fsdp \
  --model_name models/Qwen3-VL-8B-Instruct \
  --train_jsonl data/unified/train.jsonl \
  --val_jsonl data/unified/val.jsonl \
  --image_root data/raw \
  --index_dir indices \
  --bf16 \
  --batch_size 1 \
  --grad_accum 2
```

## 5c) Train with DeepSpeed ZeRO-3 (8x V100)

```bash
deepspeed --num_gpus=8 scripts/train_r3.py \
  --backend deepspeed \
  --deepspeed_config configs/deepspeed_zero3.json \
  --model_name models/Qwen3-VL-8B-Instruct \
  --train_jsonl data/unified/train.jsonl \
  --val_jsonl data/unified/val.jsonl \
  --image_root data/raw \
  --index_dir indices \
  --bf16 \
  --batch_size 1 \
  --grad_accum 2
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
  --index_dir indices
```

## Multi-GPU / Model-Parallel Notes (8x V100)

- **Full-precision fine-tuning of Qwen3-VL-8B is likely to OOM on 32GB without sharding.**
- Current training loop is **single-process** and does **not** implement FSDP/ZeRO.
- For true model-parallel training, you will need one of:
  - **FSDP (torch.distributed.fsdp)** with auto-wrap + sharded optimizer states
  - **DeepSpeed ZeRO-3** integrated into the training loop

If you want, I can add FSDP/ZeRO support and a `torchrun` command path.
