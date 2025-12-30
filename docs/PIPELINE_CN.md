# R3++ 全流程实验管线（下载 → 预处理 → 索引 → 训练 → 评测）

本文档为中文版实验操作手册，覆盖数据下载、OCR、统一数据格式、索引构建、分阶段训练和评测（含消融与不同腐蚀强度）。
默认基座模型与检索模型已在 `models/` 下。

---

## 0) 环境与依赖

```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets pillow sentence-transformers peft
pip install huggingface-hub
pip install faiss-gpu  # 或 faiss-cpu
```

如需更快下载 Hugging Face 数据集：
```bash
export HF_TOKEN=你的token
```

---

## 0.5) 下载检索模型（CLIP + 文本编码器）

```bash
python scripts/download_retrieval_models.py --out_dir models
```

默认输出：
- `models/clip-vit-b32-laion2B`
- `models/all-MiniLM-L6-v2`

---

## 1) 下载原始数据

```bash
python scripts/download_screenqa.py --out_dir data/raw/screenqa
python scripts/download_chartqa.py --out_dir data/raw/chartqa
python scripts/download_infographicvqa.py --out_dir data/raw/infovqa
```

输出结构：
- `data/raw/<dataset>/images/*.png`
- `data/raw/<dataset>/*_raw_<split>.jsonl`

---

## 2) OCR 缓存（推荐，用于高质量 pseudo-text）

建议用 PaddleOCR（速度较好，输出更稳定）：
```bash
pip install paddleocr
```

单机：
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

多卡并行 OCR（8x V100）：
```bash
torchrun --nproc_per_node=8 scripts/build_ocr_cache.py \
  --raw_dir data/raw/screenqa \
  --image_root data/raw/screenqa \
  --out_path data/ocr/screenqa_ocr.jsonl \
  --engine paddleocr \
  --lang en \
  --use_angle_cls \
  --use_gpu \
  --num_shards 8 \
  --bind_gpu

python scripts/build_ocr_cache.py --out_path data/ocr/screenqa_ocr.jsonl --merge_shards 8
```

---

## 3) 构建统一格式 JSONL（含 OCR）

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

统一 `image_path` 会带前缀，如：
- `screenqa/images/train_0.png`
- `chartqa/images/validation_123.png`
- `infovqa/images/test_4.png`

训练/评测时设置 `image_root=data/raw` 即可共享根路径。

---

## 4) 合并训练集与验证集

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

---

## 5) 构建检索索引（FAISS）

单机：
```bash
python scripts/build_indices.py \
  --jsonl data/unified/train.jsonl \
  --image_root data/raw \
  --out_dir indices \
  --text_field ocr
```

多卡分片（8x V100）：
```bash
torchrun --nproc_per_node=8 scripts/build_indices.py \
  --jsonl data/unified/train.jsonl \
  --image_root data/raw \
  --out_dir indices \
  --num_shards 8 \
  --batch_size 32 \
  --text_field ocr

python scripts/build_indices.py --out_dir indices --merge_shards 8
```

---

## 6) 训练（推荐分阶段）

### 训练说明（方法设计）
- R3++ 采用 **腐蚀模拟 + 双通道检索 + 三路径重建**。
- 训练采用 **clean → warm-up → joint → 强腐蚀** 的课程学习。
- Stage5 使用 **EMA Teacher**，避免复制完整教师导致 OOM。
- 建议 `max_length=2048`；`4096` 在多卡下可能引发 NCCL 超时或卡住。

### Stage 1：纯净 LoRA 适配（无检索、无腐蚀）
```bash
torchrun --nproc_per_node=8 scripts/train_r3.py \
  --backend fsdp \
  --model_name models/Qwen3-VL-8B-Instruct \
  --train_jsonl data/unified/train.jsonl \
  --val_jsonl data/unified/val.jsonl \
  --image_root data/raw \
  --index_dir indices \
  --output_dir checkpoints/stage1 \
  --fp16 --use_lora --disable_teacher \
  --batch_size 2 --grad_accum 4 \
  --learning_rate 1e-6 \
  --max_length 2048 --max_context_chars 128 \
  --max_steps 1000 --save_every 1000 --eval_every 1000000 \
  --sample_every 0 --num_workers 0 \
  --disable_corruption --disable_text_retrieval --disable_image_retrieval \
  --disable_prefix --disable_memory --disable_visual_memory --disable_latent_tokens \
  --disable_gate --disable_context \
  --retrieval_align_weight 0 --gate_conf_weight 0 --gate_entropy_weight 0 \
  --r3_fp32
```

### Stage 2：文本检索 warm-up（文本路径）
```bash
torchrun --nproc_per_node=8 scripts/train_r3.py \
  --backend fsdp \
  --model_name models/Qwen3-VL-8B-Instruct \
  --train_jsonl data/unified/train.jsonl \
  --val_jsonl data/unified/val.jsonl \
  --image_root data/raw \
  --index_dir indices \
  --output_dir checkpoints/stage2_text \
  --resume_from checkpoints/stage1/step_1000 \
  --fp16 --use_lora --disable_teacher \
  --batch_size 2 --grad_accum 4 \
  --learning_rate 1e-6 \
  --max_length 2048 --max_context_chars 128 \
  --max_steps 1000 --save_every 1000 --eval_every 1000000 \
  --sample_every 0 --num_workers 0 \
  --disable_corruption \
  --disable_image_retrieval --disable_visual_memory \
  --disable_gate --disable_context --disable_latent_tokens \
  --retrieval_align_weight 0.05 --retrieval_align_temperature 0.1 \
  --r3_lr_mult 0.5 \
  --r3_fp32
```

### Stage 3：图像检索 warm-up（视觉记忆）
```bash
torchrun --nproc_per_node=8 scripts/train_r3.py \
  --backend fsdp \
  --model_name models/Qwen3-VL-8B-Instruct \
  --train_jsonl data/unified/train.jsonl \
  --val_jsonl data/unified/val.jsonl \
  --image_root data/raw \
  --index_dir indices \
  --output_dir checkpoints/stage3_image \
  --resume_from checkpoints/stage2_text/step_1000 \
  --fp16 --use_lora --disable_teacher \
  --batch_size 2 --grad_accum 4 \
  --learning_rate 1e-6 \
  --loss_scale 32 \
  --r3_lr_mult 0.2 \
  --max_grad_norm 0.5 \
  --max_length 2048 --max_context_chars 128 \
  --max_steps 1500 --save_every 1500 --eval_every 1000000 \
  --sample_every 0 --num_workers 0 \
  --disable_corruption \
  --disable_text_retrieval --disable_prefix \
  --disable_memory --disable_gate --disable_context \
  --disable_latent_tokens \
  --retrieval_align_weight 0 --gate_conf_weight 0 --gate_entropy_weight 0 \
  --visual_memory_len 1 \
  --r3_fp32
```

### Stage 4：clean 联合训练
```bash
torchrun --nproc_per_node=8 scripts/train_r3.py \
  --backend fsdp \
  --model_name models/Qwen3-VL-8B-Instruct \
  --train_jsonl data/unified/train.jsonl \
  --val_jsonl data/unified/val.jsonl \
  --image_root data/raw \
  --index_dir indices \
  --output_dir checkpoints/stage4_joint \
  --resume_from checkpoints/stage3_image/step_1500 \
  --fp16 --use_lora --disable_teacher \
  --batch_size 2 --grad_accum 4 \
  --learning_rate 1e-6 \
  --max_length 2048 --max_context_chars 128 \
  --max_steps 2000 --save_every 2000 --eval_every 1000000 \
  --sample_every 0 --num_workers 0 \
  --disable_corruption \
  --disable_latent_tokens \
  --gate_conf_weight 0.1 --gate_entropy_weight 0.01 \
  --retrieval_align_weight 0.05 --retrieval_align_temperature 0.07 \
  --r3_fp32
```

### Stage 5：全量训练（强腐蚀 + EMA Teacher）
```bash
torchrun --nproc_per_node=8 scripts/train_r3.py \
  --backend fsdp \
  --model_name models/Qwen3-VL-8B-Instruct \
  --train_jsonl data/unified/train.jsonl \
  --val_jsonl data/unified/val.jsonl \
  --image_root data/raw \
  --index_dir indices \
  --output_dir checkpoints/stage5_full \
  --resume_from checkpoints/stage4_joint/step_2000 \
  --fp16 --use_lora \
  --batch_size 1 --grad_accum 4 \
  --learning_rate 1e-6 \
  --max_length 2048 --max_context_chars 128 \
  --max_steps 10000 --save_every 2000 --eval_every 1000 \
  --sample_every 0 --num_workers 0 \
  --gate_conf_weight 0.1 --gate_entropy_weight 0.01 \
  --retrieval_align_weight 0.05 --retrieval_align_temperature 0.07 \
  --max_corruption 1 \
  --corruption_warmup_steps 4000 \
  --corruption_total_steps 10000 \
  --teacher_mode ema \
  --teacher_ema_decay 0.999 \
  --teacher_ema_update_steps 1 \
  --teacher_ema_start_step 0 \
  --r3_fp32
```

---

## 6b) 后台训练命令（nohup）

```bash
mkdir -p logs

nohup torchrun --nproc_per_node=8 scripts/train_r3.py \
  --backend fsdp \
  --model_name models/Qwen3-VL-8B-Instruct \
  --train_jsonl data/unified/train.jsonl \
  --val_jsonl data/unified/val.jsonl \
  --image_root data/raw \
  --index_dir indices \
  --output_dir checkpoints/stage1 \
  --fp16 --use_lora --disable_teacher \
  --batch_size 2 --grad_accum 4 \
  --learning_rate 1e-6 \
  --max_length 2048 --max_context_chars 128 \
  --max_steps 1000 --save_every 1000 --eval_every 1000000 \
  --sample_every 0 --num_workers 0 \
  --disable_corruption --disable_text_retrieval --disable_image_retrieval \
  --disable_prefix --disable_memory --disable_visual_memory --disable_latent_tokens \
  --disable_gate --disable_context \
  --retrieval_align_weight 0 --gate_conf_weight 0 --gate_entropy_weight 0 \
  --r3_fp32 \
  > logs/stage1.log 2>&1 &

nohup torchrun --nproc_per_node=8 scripts/train_r3.py \
  --backend fsdp \
  --model_name models/Qwen3-VL-8B-Instruct \
  --train_jsonl data/unified/train.jsonl \
  --val_jsonl data/unified/val.jsonl \
  --image_root data/raw \
  --index_dir indices \
  --output_dir checkpoints/stage2_text \
  --resume_from checkpoints/stage1/step_1000 \
  --fp16 --use_lora --disable_teacher \
  --batch_size 2 --grad_accum 4 \
  --learning_rate 1e-6 \
  --max_length 2048 --max_context_chars 128 \
  --max_steps 1000 --save_every 1000 --eval_every 1000000 \
  --sample_every 0 --num_workers 0 \
  --disable_corruption \
  --disable_image_retrieval --disable_visual_memory \
  --disable_gate --disable_context --disable_latent_tokens \
  --retrieval_align_weight 0.05 --retrieval_align_temperature 0.1 \
  --r3_lr_mult 0.5 \
  --r3_fp32 \
  > logs/stage2_text.log 2>&1 &

nohup torchrun --nproc_per_node=8 scripts/train_r3.py \
  --backend fsdp \
  --model_name models/Qwen3-VL-8B-Instruct \
  --train_jsonl data/unified/train.jsonl \
  --val_jsonl data/unified/val.jsonl \
  --image_root data/raw \
  --index_dir indices \
  --output_dir checkpoints/stage3_image \
  --resume_from checkpoints/stage2_text/step_1000 \
  --fp16 --use_lora --disable_teacher \
  --batch_size 2 --grad_accum 4 \
  --learning_rate 1e-6 \
  --loss_scale 32 \
  --r3_lr_mult 0.2 \
  --max_grad_norm 0.5 \
  --max_length 2048 --max_context_chars 128 \
  --max_steps 1500 --save_every 1500 --eval_every 1000000 \
  --sample_every 0 --num_workers 0 \
  --disable_corruption \
  --disable_text_retrieval --disable_prefix \
  --disable_memory --disable_gate --disable_context \
  --disable_latent_tokens \
  --retrieval_align_weight 0 --gate_conf_weight 0 --gate_entropy_weight 0 \
  --visual_memory_len 1 \
  --r3_fp32 \
  > logs/stage3_image.log 2>&1 &

nohup torchrun --nproc_per_node=8 scripts/train_r3.py \
  --backend fsdp \
  --model_name models/Qwen3-VL-8B-Instruct \
  --train_jsonl data/unified/train.jsonl \
  --val_jsonl data/unified/val.jsonl \
  --image_root data/raw \
  --index_dir indices \
  --output_dir checkpoints/stage4_joint \
  --resume_from checkpoints/stage3_image/step_1500 \
  --fp16 --use_lora --disable_teacher \
  --batch_size 2 --grad_accum 4 \
  --learning_rate 1e-6 \
  --max_length 2048 --max_context_chars 128 \
  --max_steps 2000 --save_every 2000 --eval_every 1000000 \
  --sample_every 0 --num_workers 0 \
  --disable_corruption \
  --disable_latent_tokens \
  --gate_conf_weight 0.1 --gate_entropy_weight 0.01 \
  --retrieval_align_weight 0.05 --retrieval_align_temperature 0.07 \
  --r3_fp32 \
  > logs/stage4_joint.log 2>&1 &

nohup torchrun --nproc_per_node=8 scripts/train_r3.py \
  --backend fsdp \
  --model_name models/Qwen3-VL-8B-Instruct \
  --train_jsonl data/unified/train.jsonl \
  --val_jsonl data/unified/val.jsonl \
  --image_root data/raw \
  --index_dir indices \
  --output_dir checkpoints/stage5_full \
  --resume_from checkpoints/stage4_joint/step_2000 \
  --fp16 --use_lora \
  --batch_size 2 --grad_accum 4 \
  --learning_rate 1e-6 \
  --max_length 2048 --max_context_chars 128 \
  --max_steps 10001 --save_every 2000 --eval_every 100000 \
  --sample_every 0 --num_workers 0 \
  --gate_conf_weight 0.1 --gate_entropy_weight 0.01 \
  --retrieval_align_weight 0.05 --retrieval_align_temperature 0.07 \
  --max_corruption 1 \
  --corruption_warmup_steps 5000 \
  --corruption_total_steps 10000 \
  --teacher_mode ema \
  --teacher_ema_decay 0.999 \
  --teacher_ema_update_steps 1 \
  --teacher_ema_start_step 0 \
  --r3_fp32 \
  > logs/stage5_full.log 2>&1 &
```

查看训练日志：
```bash
tail -f logs/stage5_full.log
```

---

## 7) 评测（基模 / R3 / 消融 / 分数据集）

评测阶段默认会打印进度日志（每个腐蚀等级的 batch 进度与 ETA）。
如果需要更频繁的日志，可加 `--eval_log_every`。
如需定期打印样例（数据详情+预测+检索），可加：
`--sample_every N --sample_max K`。也可通过 `--batch_size` 提升评测吞吐。
下方所有评测/消融命令都支持多卡：把 `python` 替换为
`torchrun --nproc_per_node=8` 即可。

### 7.1 基模（Clean + Corrupt）
```bash
python scripts/eval_r3.py \
  --eval_mode base \
  --clean_only \
  --no_pseudo_text \
  --model_name models/Qwen3-VL-8B-Instruct \
  --val_jsonl data/unified/val.jsonl \
  --image_root data/raw \
  --index_dir indices \
  --top_k 3 \
  --batch_size 2 \
  --eval_log_every 50 \
  --sample_every 200 \
  --sample_max 5
```

### 多卡评测（更快）
```bash
torchrun --nproc_per_node=8 scripts/eval_r3.py \
  --eval_mode r3 \
  --checkpoint_dir checkpoints/step_1000 \
  --model_name models/Qwen3-VL-8B-Instruct \
  --val_jsonl data/unified/val.jsonl \
  --image_root data/raw \
  --index_dir indices \
  --top_k 3 \
  --batch_size 2 \
  --eval_log_every 50 \
  --sample_every 200 \
  --sample_max 5
```

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

### 7.2 基模（按数据集评测）
```bash
python scripts/eval_r3.py \
  --eval_mode base \
  --clean_only \
  --no_pseudo_text \
  --dataset_prefixes screenqa,chartqa,infovqa \
  --model_name models/Qwen3-VL-8B-Instruct \
  --val_jsonl data/unified/val.jsonl \
  --image_root data/raw \
  --index_dir indices \
  --top_k 3 \
  --out_json results/base_clean_by_dataset.json
```

```bash
python scripts/eval_r3.py \
  --eval_mode base \
  --corruption_levels 0,0.2,0.4,0.6,0.8 \
  --no_pseudo_text \
  --dataset_prefixes screenqa,chartqa,infovqa \
  --model_name models/Qwen3-VL-8B-Instruct \
  --val_jsonl data/unified/val.jsonl \
  --image_root data/raw \
  --index_dir indices \
  --top_k 3 \
  --out_json results/base_corrupt_by_dataset.json
```

### 7.3 R3 模型（Clean + Corrupt）
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

### 7.4 R3 按数据集评测
```bash
python scripts/eval_r3.py \
  --eval_mode r3 \
  --clean_only \
  --dataset_prefixes screenqa,chartqa,infovqa \
  --checkpoint_dir checkpoints/step_1000 \
  --model_name models/Qwen3-VL-8B-Instruct \
  --val_jsonl data/unified/val.jsonl \
  --image_root data/raw \
  --index_dir indices \
  --top_k 3 \
  --out_json results/r3_clean_by_dataset.json
```

```bash
python scripts/eval_r3.py \
  --eval_mode r3 \
  --corruption_levels 0,0.2,0.4,0.6,0.8 \
  --dataset_prefixes screenqa,chartqa,infovqa \
  --checkpoint_dir checkpoints/step_1000 \
  --model_name models/Qwen3-VL-8B-Instruct \
  --val_jsonl data/unified/val.jsonl \
  --image_root data/raw \
  --index_dir indices \
  --top_k 3 \
  --out_json results/r3_corrupt_by_dataset.json
```

### 7.5 R3 消融评测
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

```bash
python scripts/eval_r3.py \
  --eval_mode r3 \
  --clean_only \
  --disable_visual_memory \
  --disable_latent_tokens \
  --checkpoint_dir checkpoints/step_1000 \
  --model_name models/Qwen3-VL-8B-Instruct \
  --val_jsonl data/unified/val.jsonl \
  --image_root data/raw \
  --index_dir indices \
  --top_k 3
```

---

## 8) 实验设计与评估方法说明（文本版）

- 数据集：ScreenQA、ChartQA、InfographicVQA。
- 目标：在 **部分模态缺失/腐蚀** 条件下做生成式多模态 QA。
- 方法：R3++ = 腐蚀模拟 + 文本/图像双通道检索 + 三路径重建 + 自适应门控 + clean/corrupt 一致性。
- 训练策略：clean → 单通道 warm-up → clean joint → 强腐蚀 + EMA teacher。
- 评测：在 `corruption_levels` 的不同强度下评测，报告 EM/F1/BLEU/ROUGE（代码内已实现）。如需 ANLS 等指标可后续扩展。
