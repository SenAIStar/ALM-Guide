# SeamlessM4T v2 语音翻译

## 项目定位

系统同时保留 ASR + MT + TTS 级联 Baseline 和 SeamlessM4T v2 直接 S2ST 推理。级联链路便于定位转写、翻译和合成各自的错误；直接路线用于比较整体语义、韵律和延迟。

当前训练入口微调的是 `SeamlessM4Tv2ForSpeechToText` 的语义翻译路径，生成目标文本标签，并通过 LoRA 控制训练参数量。直接 S2ST 推理继续复用 checkpoint 的 text-to-unit 模块与声码器。代码没有声称覆盖 Meta 的完整 UnitY2 训练配方，也没有训练语音单元解码器或声码器。

## 数据协议

JSONL 每行至少包含源音频、源/目标语言代码、源转写和目标翻译。语言代码必须使用 checkpoint 支持的 SeamlessM4T 代码；数据集和模型许可证要在商用前单独核对。

```json
{"audio_path":"audio/0001.wav","source_language":"cmn","target_language":"eng","source_text":"会议推迟到周五。","target_text":"The meeting is postponed to Friday."}
```

## 运行

```powershell
accelerate launch train_adapter.py --config config.json --manifest data/train.jsonl --validation-manifest data/validation.jsonl
python infer.py --config config.json --audio source.wav --output translated.wav
python evaluate.py --records eval.jsonl
```

训练前先用 dedicated S2TT 模型验证标签和 loss，再用顶层 `SeamlessM4Tv2Model` 做 S2ST 推理。这样能把语义路径和语音生成路径的故障分开。

## 评测

目标语音先用固定 ASR 转写，再计算 BLEU/chrF 与 ASR-WER；同时报告语音自然度、说话人/韵律保持和端到端延迟。级联 Baseline 要保存中间 ASR 和 MT 输出，直接模型要保存中间文本 token，便于归因。

建议 ablation：级联与直接 S2ST；零样本与 S2TT LoRA；完整句与分块推理；不同 chunk overlap；是否保留说话人条件。当前结果状态为 `not_measured`。

## 常见故障

- 输出语言不对：核对 `tgt_lang` 和训练标签中的语言 token。
- 翻译文本正确但语音差：故障位于 text-to-unit 或声码器，不应归因给 S2TT LoRA。
- 分块重复或漏译：对齐 chunk 边界、中间文本和音频拼接，不能只调 overlap。
- 显存超限：优先使用 dedicated S2TT 训练类、LoRA 和梯度累积。
