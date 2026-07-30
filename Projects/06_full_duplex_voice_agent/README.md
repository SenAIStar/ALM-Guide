# 全双工语音 Agent

## 项目定位

重点不是把 ASR、LLM、TTS 顺序串联，而是同时处理用户上行语音和助手下行语音。状态机区分 idle、listening、thinking、speaking、cancelling；每次生成带 `generation_id`，打断后必须取消对应任务并清空同 id 的未播放音频。

## 数据协议

服务端事件使用 JSONL 记录 `session_id`、`event`、`timestamp_ms` 和可选的 `generation_id`。时间戳来自单调时钟。用于评测的关键事件包括 `speech_start`、`endpoint`、`first_audio_chunk`、`barge_in_detected`、`generation_cancelled` 和 `playback_drained`。

```json
{"session_id":"s1","event":"barge_in_detected","timestamp_ms":10420,"generation_id":"g7"}
```

WebSocket 消息只承载状态事件；真实 ASR、LLM 流、TTS chunk、VAD 和工具路由需要接入 `server.py` 的对应分支。当前实现用于验证状态迁移与 stale cancellation，不伪装成完整生产服务。

## 运行

```powershell
uvicorn server:app --host 0.0.0.0 --port 8000
python evaluate.py --events logs/session.jsonl
python -m unittest discover -s tests -v
```

## 评测

至少报告用户端点到首个音频 chunk、打断检测、取消完成和播放队列清空四段延迟，并同时给出误打断率、漏打断率和工具调用正确率。均值不足以描述实时系统，正式报告需要 P50/P95/P99。

建议 ablation：不同 VAD 阈值；回声消除前后；硬取消与软淡出；有无 backchannel；串行与并行 ASR/LLM。当前真实时延状态为 `not_measured`。

## 常见故障

- 旧音频在打断后继续播放：队列项缺少 `generation_id` 或取消时只停模型、未清播放缓存。
- 助手声音触发用户 VAD：检查 AEC、播放参考信号和双讲检测。
- stale cancellation 取消新回答：所有取消与完成事件都要校验 generation id。
- 工具调用阻塞语音：工具任务与音频流分离，并设置可追踪的超时与降级。
