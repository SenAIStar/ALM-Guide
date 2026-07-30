# 语音大模型项目

这里放的是 7 个相互独立的语音项目。前 3 个解决 ASR、TTS 和语音翻译的基础微调问题；中间 3 个覆盖听觉指令、长音频定位和全双工交互；最后一个项目用 Qwen2.5-Omni 做全模态基线与反事实评测。

| 编号 | 项目 | 训练对象 | 主要评测 |
| --- | --- | --- | --- |
| 01 | [Whisper 低资源 ASR](01_whisper_low_resource_asr/) | Whisper 或 LoRA | WER、CER、长音频切片 |
| 02 | [SpeechT5 多说话人 TTS](02_speecht5_multispeaker_tts/) | SpeechT5 声学模型 | ASR-WER、说话人相似度、F0、RTF |
| 03 | [SeamlessM4T v2 语音翻译](03_seamless_s2st/) | S2TT 语义路径 LoRA | BLEU、ASR-WER、延迟、错误传播 |
| 04 | [Audio LLM 指令微调](04_audio_llm_instruction_tuning/) | Connector 与 LLM LoRA | 分任务准确率、F1、证据一致性 |
| 05 | [长音频定位问答](05_long_audio_grounding/) | 定位、说话人和答案头 | temporal IoU、Recall@K、grounded accuracy |
| 06 | [全双工语音 Agent](06_full_duplex_voice_agent/) | 状态机与流式服务 | 首包、打断、取消、误打断率 |
| 07 | [全模态语音助手](07_qwen_omni_assistant/) | 冻结 Omni 基线与投影层 ablation | 模态移除、冲突偏好、生成延迟 |

## 环境

每个项目单独维护 `requirements.txt`。建议为每个目录创建虚拟环境，避免 `transformers`、CUDA 和音频依赖互相污染。模型权重与数据集不会随代码分发，第一次运行需要联网下载，并遵守模型、数据集和语音授权条款。

```powershell
cd 01_whisper_low_resource_asr
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## 验证

根目录验证脚本会编译全部 Python 文件、检查 7 份配置与 README，并运行不依赖模型权重的单元测试。

```powershell
python -B -X utf8 verify.py
```

这一步只证明数据协议、指标函数和状态逻辑能工作，不等于模型训练已经完成。所有配置默认保留 `metrics_status: not_measured`；WER、MOS、RTF、显存和端到端时延必须在固定数据、硬件、checkpoint revision 与解码参数下重新测量。

## 实验记录

每次实验至少保存以下信息：代码版本、配置文件、训练与验证清单哈希、模型 revision、随机种子、硬件、依赖锁文件、最优 checkpoint、逐样本预测和失败样本。只有固定测试集上的最终结果可以进入简历。
