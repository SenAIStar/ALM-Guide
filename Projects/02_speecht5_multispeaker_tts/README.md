# SpeechT5 多说话人 TTS

## 项目定位

SpeechT5 预测 Mel 频谱，HiFi-GAN 把 Mel 转成波形。项目重点是多说话人条件、文本到声学特征的对齐和稳定评测，不把单条试听当成效果结论。Speaker Encoder、X-vector 归一化和声码器版本都要固定。

## 数据协议

JSONL 每行包含 `audio_path`、`text`、`speaker_id`、`speaker_embedding_path`、`duration`。当前 checkpoint 的 speaker embedding 维度是 512；替换 Speaker Encoder 时必须重新生成全量 embedding，并核对训练与推理的 L2 归一化方式。

```json
{"audio_path":"audio/spk1_0001.wav","text":"今天开始联调。","speaker_id":"spk_001","speaker_embedding_path":"emb/spk_001.json","duration":2.1}
```

数据切分按说话人目标决定：做已见说话人合成时，同一说话人可以跨 split，但句子不能重复；做零样本声音迁移时，测试说话人必须完全隔离。静音、削波、错字和极端长句应提前过滤。

## 运行

```powershell
accelerate launch train.py --config config.json
python synthesize.py --config config.json --text "语音合成基线" --speaker embedding.json --output demo.wav
python evaluate.py --records eval_records.jsonl
```

`train.py` 关闭 `use_cache`，使用 SpeechT5 processor 生成 Mel 标签。训练前先用 20 条样本过拟合，确认 stop token、reduction factor、padding 和 speaker embedding 都能被模型读取。

## 评测

自动评测拆成 ASR-WER/CER、speaker similarity、F0 RMSE、时长偏差和 RTF；主观测试至少区分自然度、可懂度和说话人相似度。MOS 必须盲测并记录听众数量、耳机环境和打分协议。

建议 ablation：单说话人与多说话人；原始与平均 speaker embedding；冻结 Encoder、冻结 Decoder 和全量训练；预训练与域内声码器。当前结果状态为 `not_measured`。

## 常见故障

- 合成内容可懂但音色漂移：核对 embedding 来源、维度和归一化。
- 尾部重复或提前停止：检查 stop labels、reduction factor 和长句分布。
- Mel 正常但波形失真：把声学模型和声码器分开验收，确认 Mel 标度匹配。
- 训练 loss 正常但试听差：先听固定验证集，再看对齐图和逐音素错误，不要只盯总 loss。
