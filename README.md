<div align="center">
  <img src="assets/xiaosen-ai-logo.png" width="128" alt="小森学AI Logo" />

# ALM-Guide

从语音信号到全双工交互，面向语音多模态算法岗的学习、面试与项目代码

</div>

## 关于这个仓库

一个能完成语音识别或语音合成的 Demo，距离可用的语音系统还有很长一段路。采样率、特征提取、说话人差异、噪声、流式延迟、打断响应、工具调用和隐私安全，都会直接影响最终体验。

这个仓库围绕语音多模态算法岗需要掌握的知识整理代码，内容分为讲义、面试题和项目三部分。重点不是堆模型名称，而是把音频怎样进入模型、怎样训练、怎样评测以及怎样定位问题讲清楚。

## 内容结构

### 1. LectureNotes：大模型讲义代码

[LectureNotes](LectureNotes/) 对应语音多模态讲义中的核心代码，主要包括：

- 音频基础：采样、分帧、STFT、Fbank、MFCC 和常见数据处理；
- 语音表征：wav2vec 2.0、HuBERT、data2vec、Transformer、Conformer；
- 音频离散化：Codec、VQ、RVQ 及语音 Token；
- ASR：CTC、RNN-T、AED、Whisper、Paraformer、SenseVoice、Qwen3-ASR；
- TTS：声学模型、Vocoder、Tacotron、FastSpeech、VITS、VALL-E、扩散式 TTS、CosyVoice、Qwen3-TTS；
- SpeechLM 与 Omni：语音语言模型、端到端语音交互、流式与全双工系统、后训练和评测。

### 2. InterviewQA：大模型面试题代码实现

[InterviewQA](InterviewQA/) 面向语音算法与语音多模态岗位，内容会覆盖语音前端、增强与分离、ASR、TTS、SpeechLM、级联系统与端到端系统、全双工交互、数据训练、评测部署和安全问题。

代码部分会选择适合手写或实验验证的题目展开，并补充输入输出、复杂度、适用条件和常见错误。

### 3. Projects：语音多模态项目

[Projects](Projects/) 计划补充 7 个项目：

1. 低资源语音识别（Whisper + Common Voice）；
2. 多说话人语音合成（SpeechT5 + HiFi-GAN）；
3. 端到端语音翻译（SeamlessM4T v2）；
4. Audio LLM 听觉指令微调；
5. 长音频多说话人时间定位与问答；
6. 全双工语音 Agent；
7. 端到端 Omni 助手。

项目会围绕数据准备、训练流程、评测指标、流式推理和错误分析组织。当前仓库已经搭好目录，代码正在按项目补充。

## 怎么使用

如果你的语音基础比较薄弱，建议先从 `LectureNotes` 中的信号处理、声学特征和 ASR/TTS 基础开始，再学习 Codec、SpeechLM 和 Omni。

如果你正在准备面试，可以用 `InterviewQA` 梳理原理，再从 `Projects` 中选择一到两个方向完成实验。简历里只写自己真正做过的部分，并保留数据规模、硬件配置、评测口径和失败案例。

## 代码说明

- 仓库关注核心算法与实验流程，不提供可直接上线的完整语音服务；
- 当前项目代码仍在补充，项目目录和说明不能等同于已经完成的实验结果；
- 示例数据和模拟输出只用于说明流程，不能写成真实线上指标；
- 使用语音数据时，请确认数据集许可证、说话人授权、隐私处理和合成语音标识要求；
- 训练前请根据显存、音频时长和采样率调整 batch size、梯度累积和切片策略。

## 关于作者

小森，现任互联网大厂大模型算法工程师，曾在微软亚洲研究院从事算法研究工作。长期关注大模型的技术演进与工程实践，内容涉及 LLM、VLM、Diffusion、Audio、Omni 及搜索推荐等方向。

全网同名：**小森学AI**

- 小红书：[小森学AI](https://www.xiaohongshu.com/user/profile/5c5bb6f8000000001b0177fa)
- Bilibili：[小森学AI](https://space.bilibili.com/498993077)

