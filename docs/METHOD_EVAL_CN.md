# R3++ 方法与训练评估说明（中文）

本文档仅描述**研究方法、模型结构、训练策略与评估方案**，不包含具体命令。与仓库代码保持一致。

---

## 1. 研究目标与问题定义

R3++（Retrieval‑Augmented Reconstruction for Robust Multimodal QA under Partial Modality Corruption）旨在提升多模态 QA 在**部分模态缺失/损坏**场景下的鲁棒性与可靠性。  
在图像或文本被遮挡、模糊、裁剪、OCR 缺失等条件下，模型仍应生成准确答案，并降低语言偏置导致的幻觉。

给定输入图像 \(I\)、问题 \(q\)、伪文本 \(t\)（OCR/字幕/布局等占位信息），在腐蚀条件下目标建模：
\[
p(a \mid \tilde{I}, \tilde{q}, \tilde{t}, E)
\]
其中 \(E\) 为检索到的文本/图像证据。

---

## 2. 模型结构概览

R3++ 由四个核心模块组成：

1. **不确定性感知腐蚀模拟器**  
   - 视觉端：模糊、遮挡、裁剪、随机擦除、噪声、JPEG 失真、降采样、颜色扰动、运动模糊等。  
   - 文本端：截断、字符噪声、token 丢弃。  
   - 输出：腐蚀后的输入 \(\tilde{I}, \tilde{q}, \tilde{t}\) 与置信图 \(c_{\text{vis}}, c_{\text{text}}\)。

2. **双通道检索（文本 + 图像）**  
   - 文本检索：用 sentence‑transformers 编码伪文本索引（OCR/伪文本），FAISS 检索 top‑k。  
   - 图像检索：用 CLIP 视觉编码构建索引，FAISS 检索 top‑k 图像证据。  
   - 融合：将文本与图像证据合并，供后续重建与门控。

3. **三路径重建（Selective Reconstruction）**  
   - **PrefixEnhancer**：对检索文本做摘要/压缩，生成软前缀向量。  
   - **MemoryAligner**：把检索到的文本/图像嵌入对齐到统一记忆空间。  
   - **LatentImputationTokens**：在低置信区域注入可学习的缺失补全 token。  
   - **AdaptiveGate**：基于置信图在视觉特征与检索记忆之间自适应融合。

4. **教师‑学生一致性（Clean vs Corrupt）**  
   - 学生模型处理腐蚀输入与检索证据；  
   - 教师模型处理干净输入（不腐蚀），用于一致性约束；  
   - 教师采用 **EMA Teacher**（避免复制完整模型导致显存爆炸）。

底座模型为 **Qwen3‑VL‑8B**；训练默认 FP16/bfloat16；LoRA 可选且默认启用。

---

## 3. 损失函数设计

训练目标由多项损失组成：

1. **生成式交叉熵损失**  
   \[
   \mathcal{L}_{\text{CE}} = -\log p_{\theta}(a \mid \tilde{I},\tilde{q},\tilde{t}, E)
   \]

2. **一致性损失（教师‑学生）**  
   采用 token 级 KL 或分布对齐：
   \[
   \mathcal{L}_{\text{cons}} = \mathrm{KL}(p_{\theta} \,\|\, p_{\theta^-})
   \]
   实现中按有效 token 数归一化，避免超长序列导致损失爆炸。

3. **门控正则**  
   - **置信对齐项（gconf）**：鼓励门控与置信图一致  
   - **熵正则（gent）**：避免门控塌缩或过度极端

4. **检索对齐损失（ralign）**  
   在检索嵌入与主干视觉/文本表示间对齐，提升检索信息可用性。

总损失：
\[
\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda_{\text{cons}}\mathcal{L}_{\text{cons}}
+ \lambda_{\text{conf}}\mathcal{L}_{\text{conf}}
+ \lambda_{\text{ent}}\mathcal{L}_{\text{ent}}
+ \lambda_{\text{align}}\mathcal{L}_{\text{align}}
\]

---

## 4. 训练策略（分阶段 + 课程学习）

为确保稳定性与可控收敛，采用阶段式训练：

1. **Stage1 Clean 适配**  
   仅训练 LoRA（无检索、无腐蚀），让基座适配任务形式。

2. **Stage2 文本检索 warm‑up**  
   仅启用文本检索与文本路径重建，稳定对齐。

3. **Stage3 图像检索 warm‑up**  
   仅启用视觉检索与视觉记忆路径，避免新模块梯度爆炸。

4. **Stage4 clean 联合训练**  
   同时启用文本与图像检索，仍保持输入干净。

5. **Stage5 强腐蚀训练**  
   打开腐蚀模拟器，逐步提升腐蚀强度（max_corruption），  
   同时启用 EMA Teacher 一致性约束。

**腐蚀调度**：腐蚀强度从 0 线性或分段上升至 max_corruption，用于模拟真实缺失模态场景。

---

## 5. 评测方案

评测目标：
1. 干净输入性能  
2. 不同腐蚀强度的鲁棒性  
3. 模块消融贡献  
4. 基模 vs R3++ 对比

### 5.1 评测设置

- 评测腐蚀等级：\([0.0, 0.2, 0.4, 0.6, 0.8]\)  
- 支持基座模型 (Qwen3‑VL‑8B) 与 R3++ 对比  
- 支持按数据集分别评测（ScreenQA / ChartQA / InfovQA）

### 5.2 指标

默认实现指标：
- EM（Exact Match）
- F1
- BLEU / ROUGE

可选扩展（需自行实现）：
- ANLS（DocVQA 常用）
- Relaxed Accuracy（ChartQA 常用）

### 5.3 消融实验

可评测如下消融组合：
- 关闭文本检索 / 图像检索  
- 关闭 PrefixEnhancer  
- 关闭 MemoryAligner  
- 关闭 LatentImputationTokens  
- 关闭 AdaptiveGate  
- 关闭 Soft Prefix

---

## 6. 结论性说明

R3++ 的核心贡献在于：  
1) 在部分模态缺失条件下引入外部检索证据；  
2) 通过三路径重建与自适应门控强化对信息缺失区域的恢复；  
3) 通过 clean‑corrupt 一致性减少语言幻觉；  
4) 使用阶段式训练与 EMA teacher 保证收敛稳定性。

该流程与代码保持一致，适合用于顶刊实验的标准化复现与消融分析。

