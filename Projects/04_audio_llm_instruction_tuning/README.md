# Audio LLM 指令微调

## 项目定位

架构由音频 Encoder、Connector 和指令 LLM 组成。第一阶段冻结两端，只训练 Connector，把变长声学序列池化并映射到 LLM hidden size；第二阶段再给 LLM 注意力层加 LoRA。项目必须同时覆盖 ASR、音频事件、说话人/情感、副语言信息和开放问答，否则只能算语音转写模型。

## 数据协议

JSONL 每行包含 `sample_id`、`audio_path`、`instruction`、`answer` 和 `task`。`task` 只能取 `asr`、`audio_event`、`speaker`、`emotion`、`paralinguistic`、`audio_qa`。训练时只对答案 token 计算 causal LM loss，用户指令和音频前缀统一置为 `-100`。

```json
{"sample_id":"evt_0001","audio_path":"audio/cough.wav","instruction":"判断主要声音事件。","answer":"咳嗽声。","task":"audio_event"}
```

## 运行

```powershell
python validate_data.py --manifest data/train.jsonl
accelerate launch train.py --config config.json --manifest data/train.jsonl
python evaluate.py --records outputs/predictions.jsonl
```

默认 7B LLM 需要较大显存。接口联调时可把 `llm_id` 换成小模型，但正式结果必须恢复目标 checkpoint。Connector 训练稳定后再打开 LoRA，避免两个随机或未对齐模块同时漂移。

## 评测

评测按 task 分桶，分别使用 WER/CER、分类 F1、exact match 或人工评分，再计算 task macro average。开放问答要保存证据时间段或声学线索，避免模型只靠问题先验作答。

建议 ablation：冻结与微调音频 Encoder；mean pooling 与 query pooling；单层/双层 Connector；只训 Connector 与 Connector + LoRA；去掉某类任务后的迁移变化。当前结果状态为 `not_measured`。

## 常见故障

- loss 下降但模型忽略音频：做静音、错配音频和音频移除反事实测试。
- padding 改变结果：检查声学 attention mask 到隐藏帧的下采样。
- 模型只会转写：检查任务采样比例、指令模板泄漏和非 ASR 标签质量。
- LoRA 参数为零：打印 trainable parameters，并核对目标层名。
