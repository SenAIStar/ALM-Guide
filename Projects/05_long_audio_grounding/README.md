# 长音频定位问答

## 项目定位

输入是会议、客服或播客长音频，输出同时包含答案、证据时间段和说话人。系统先用 VAD/diarization 生成候选片段，再把问题与片段编码成 question-conditioned features，定位头预测起止帧，说话人头和答案头完成归因与回答。

仓库里的 `train.py` 训练下游多任务头，输入是上游编码器预计算的融合特征。这样可以在不反复编码数小时音频的情况下验证定位损失和采样策略；它不等于已经训练完整的 VAD、diarization 或生成式问答模型。

## 数据协议

训练清单使用 JSONL，每行包含 `feature_path`、`start_index`、`end_index`、`speaker_index` 和 `answer_index`。`feature_path` 指向形状为 `[frames, hidden_size]` 的 PyTorch tensor，必须已经融合问题与对应音频窗口。

```json
{"feature_path":"features/meeting01_q03.pt","start_index":42,"end_index":87,"speaker_index":2,"answer_index":15}
```

## 运行

```powershell
python train.py --config config.json --manifest data/train.jsonl --output outputs/heads.pt
python pipeline.py --audio meeting.wav --question "谁提出了延期？"
python pipeline.py --audio meeting.wav --question "谁提出了延期？" --features features/query.pt --checkpoint outputs/heads.pt
python evaluate.py --records eval.jsonl --iou-threshold 0.5
```

没有 `--features` 时，`pipeline.py` 只输出滑窗计划；提供融合特征和 checkpoint 后才执行定位头推理。这个区分用于防止把候选窗口误写成模型预测。

定位头不会分别对起点、终点做独立 argmax，而是在 `start_index <= end_index` 的约束下联合选择得分最高的时间段。多说话人候选做 temporal NMS 时只抑制同一说话人的重叠区间，避免把同时发言错误地合并成一条证据。

## 评测

核心指标是 temporal IoU、Recall@K、speaker attribution、answer score 和 evidence-grounded accuracy。只有答案正确、说话人正确且 IoU 达标，样本才算 grounded correct。重叠说话、跨窗口证据和无答案问题要单独分桶。

建议 ablation：固定窗口与 VAD 片段；无说话人头；无问题条件；单任务与多任务 loss；不同 NMS 阈值。当前结果状态为 `not_measured`。

## 常见故障

- 答案正确但定位错：模型可能利用问题先验，必须降低该样本的 grounded score。
- 边界总贴窗口两端：检查候选窗口长度、归一化坐标和跨窗口样本。
- 说话人错配：统一 diarization 标签与训练 speaker index 映射。
- 长音频漏召回：先查 VAD/候选生成，再查定位头，避免把前级漏检归因给问答模型。
