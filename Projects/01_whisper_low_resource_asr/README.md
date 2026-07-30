# Whisper 低资源 ASR

## 项目定位

目标是把 Whisper 适配到低资源语言或垂直领域。第一版先跑冻结 checkpoint 的零样本 Baseline，再比较 LoRA、冻结 Encoder 和全量微调。所有方案共用同一份说话人隔离测试集、文本规范化规则和解码参数。

## 数据协议

清单使用 JSONL，每行包含 `audio_path`、`text`、`language`、`speaker_id`、`duration` 和 `split`。音频进入模型前会转单声道并重采样到 `config.json` 的采样率。同一 `speaker_id` 不能跨 train、validation 和 test；重复音频、空转写、异常时长和错误采样率应在制作清单时拦截。

```json
{"audio_path":"audio/0001.wav","text":"设备已恢复","language":"zh","speaker_id":"spk_001","duration":1.84,"split":"train"}
```

## 运行

```powershell
python prepare_manifest.py --input raw.jsonl --output data/train.jsonl --split train
python prepare_manifest.py --input raw_validation.jsonl --output data/validation.jsonl --split validation --check-against data/train.jsonl
python prepare_manifest.py --input raw_test.jsonl --output data/test.jsonl --split test --check-against data/train.jsonl --check-against data/validation.jsonl
accelerate launch train.py --config config.json --use-lora
python evaluate.py --reference refs.txt --hypothesis hyps.txt
python transcribe.py --config config.json --audio demo.wav
```

`--check-against` 可以重复传入已有清单。制作 validation 和 test 时必须把之前的 split 一并检查，否则只检查当前输出文件无法发现跨 split 的说话人泄漏。

`train.py` 使用 `Seq2SeqTrainer`，LoRA 默认挂在注意力 `q_proj/v_proj`。显存不足时先降低 batch size、开启梯度累积，再考虑冻结 Encoder；不要通过截短验证集来换显存。

## 评测

中文同时报告 CER 和按业务分词规则计算的 WER；数字、英文缩写和标点规范化必须在预测前冻结。除总体指标外，还要按时长、信噪比、说话人、设备和领域词分桶。长音频另报漏切、重叠切片和时间戳偏移。

建议 ablation：零样本、LoRA、冻结 Encoder、全量微调；有无 SpecAugment；beam size 与语言提示；通用词表与领域热词。当前结果状态为 `not_measured`。

## 常见故障

- 训练下降但测试不变：先查说话人泄漏、文本规范化不一致和语言 token。
- 短句正常、长句漏字：检查 30 秒切片、重叠窗口和拼接策略。
- 数字错误集中：单独定义数字规范化与逆文本归一化，不要在评分脚本里临时改规则。
- LoRA 没有梯度：打印 trainable parameters，并确认目标层名与当前模型版本一致。
