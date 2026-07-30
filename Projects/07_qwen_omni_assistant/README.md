# 全模态语音助手

## 项目定位

项目以 Qwen2.5-Omni 为冻结基线，统一处理文本、图像、视频和音频。Thinker 负责多模态理解与文本语义，Talker 负责语音输出。`use_audio_in_video` 必须在多模态解析、processor 和 `generate` 三处保持一致；不需要语音输出时可关闭 Talker。

官方 checkpoint 的完整训练配方没有封装在本仓库。`train_projector.py` 只训练离线缓存特征上的轻量投影层，用来验证模态对齐和消融流程，不会把该结果描述成 Qwen2.5-Omni 全量微调。

## 数据协议

推理/评测清单包含 `sample_id`、`modalities`、`instruction`、`answer` 和 `conflict_type`。冲突样本至少包含两种模态。投影层训练清单额外包含 `source_feature_path`、`target_feature_path` 和 `modality`，两个 tensor 分别为 `[tokens, input_size]` 与 `[tokens, hidden_size]`。

```json
{"sample_id":"av_001","modalities":["audio","image"],"instruction":"结合声音和画面判断事件。","answer":"玻璃杯掉落。","conflict_type":"none"}
```

## 运行

```powershell
python validate_data.py --manifest data/train.jsonl
python infer.py --config config.json --audio question.wav --image frame.jpg --text "结合声音和画面回答"
python train_projector.py --manifest data/projector_train.jsonl --input-size 1024 --hidden-size 3584 --output outputs/projector.pt
python evaluate.py --records eval_counterfactual.jsonl
```

`infer.py` 的 system prompt 与音频输出协议按官方模型卡设置。Windows 环境如果无法安装 `decord`，多模态预处理会回退到 torchvision，视频性能需要单独测。

## 评测

除任务正确率外，必须构造静音、遮图、去字幕、错配图像和冲突文本等反事实样本。任务要求依赖某模态时，移除该模态应产生可解释的分数下降；不相关模态被移除后不应大幅改变答案。

建议 ablation：audio/image/text removal；是否读取视频音轨；Thinker-only 与 Thinker + Talker；原始特征与训练投影层；冲突样本中的来源偏好。当前结果状态为 `not_measured`。

## 常见故障

- `KeyError: qwen2_5_omni`：检查 `transformers` 版本是否与项目 requirements 一致。
- 视频音轨行为不一致：核对 `use_audio_in_video` 三处参数。
- 关闭 Talker 后仍请求音频：`disable_talker()` 后必须使用 `return_audio=False`。
- 反事实结果无变化：模型可能依赖字幕或问题先验，需要检查输入是否真正移除了目标模态。
