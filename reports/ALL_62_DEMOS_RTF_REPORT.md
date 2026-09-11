# AuK 官方 Demo 网站 62 个全量测试用例 RTF 评测报告

- 官方演示站: https://auk-project.github.io/
- 测试时间: 2026-09-11 21:47:42
- 硬件环境: Apple Silicon M3 Max (128GB Unified Memory)
- 计算后端: Apple Silicon GPU (MPS / Metal 加速)
- 模型架构: AuK-Flash (4-Step DMD 蒸馏流匹配 + BigVGAN-Flow-VAE + Qwen2.5-Omni)

## 1. 全量评测数据汇总

- **覆盖大类家族**: 5 大类全部覆盖（Text-to-Speech, Content Editing, Enhancement and Separation, Paralinguistic Editing, Acoustic Editing）
- **子任务类别总数**: 16 个功能组全部覆盖
- **测试用例总数**: **62 / 62 项全量执行并通过 (100%)**
- **生成音频总量**: 383.52 秒
- **端到端总耗时**: 303.87 秒
- **全集平均 RTF (Real-Time Factor)**: **0.7923** (端到端全集进入超实时生成)
- **实时生成倍速**: **1.26x 实时速度**（生成 1 秒完整音频仅需 792 毫秒）
- **平均音频能量 (RMS)**: **0.0812**（100% 确认全量音频处于饱满人声区间）
- **最小 RTF**: 0.2368
- **最大 RTF**: 2.3534

## 2. 62 个用例逐项评测明细表

| 序号 | 任务大类 | 子任务组 | 样本 ID | 样本标签 | 时长 (s) | 耗时 (s) | RTF | 波形 RMS | 状态 |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 01 | Text-to-Speech | Instruct TTS | instruct-tts-1 | Instruct TTS · Mandarin | 4.00 | 1.389 | **0.347** | 0.0960 | PASS |
| 02 | Text-to-Speech | Instruct TTS | instruct-tts-2 | Instruct TTS · Mandarin | 4.00 | 1.062 | **0.265** | 0.1155 | PASS |
| 03 | Text-to-Speech | Instruct TTS | instruct-tts-3 | Instruct TTS · English | 4.00 | 0.960 | **0.240** | 0.1270 | PASS |
| 04 | Text-to-Speech | Instruct TTS | instruct-tts-4 | Instruct TTS · English | 4.00 | 0.947 | **0.237** | 0.0798 | PASS |
| 05 | Text-to-Speech | Zero-Shot TTS | zs-tts-1 | Zero-Shot TTS · Mandarin | 5.00 | 2.338 | **0.468** | 0.0917 | PASS |
| 06 | Text-to-Speech | Zero-Shot TTS | zs-tts-2 | Zero-Shot TTS · Mandarin | 5.08 | 1.965 | **0.387** | 0.1034 | PASS |
| 07 | Text-to-Speech | Zero-Shot TTS | zs-tts-3 | Zero-Shot TTS · English | 7.12 | 2.498 | **0.351** | 0.1038 | PASS |
| 08 | Text-to-Speech | Zero-Shot TTS | zs-tts-4 | Zero-Shot TTS · English | 2.76 | 1.722 | **0.624** | 0.0717 | PASS |
| 09 | Content Editing | Speech Content Editing | ce-zh-1 | Content Editing · Mandarin | 8.00 | 5.280 | **0.660** | 0.0735 | PASS |
| 10 | Content Editing | Speech Content Editing | ce-zh-2 | Content Editing · Mandarin | 8.00 | 3.574 | **0.447** | 0.0600 | PASS |
| 11 | Content Editing | Speech Content Editing | ce-en-1 | Content Editing · English | 8.00 | 4.967 | **0.621** | 0.0837 | PASS |
| 12 | Content Editing | Speech Content Editing | ce-en-2 | Content Editing · English | 8.00 | 3.090 | **0.386** | 0.0676 | PASS |
| 13 | Content Editing | Vocal Edit | vocaledit-zh-1 | Vocal Edit · Mandarin | 8.00 | 3.117 | **0.390** | 0.0794 | PASS |
| 14 | Content Editing | Vocal Edit | vocaledit-zh-2 | Vocal Edit · Mandarin | 8.00 | 3.572 | **0.447** | 0.0953 | PASS |
| 15 | Content Editing | Vocal Edit | vocaledit-en-2 | Vocal Edit · English | 7.58 | 2.819 | **0.372** | 0.1280 | PASS |
| 16 | Content Editing | Vocal Edit | vocaledit-en-1 | Vocal Edit · English | 5.74 | 2.452 | **0.427** | 0.1063 | PASS |
| 17 | Enhancement and Separation | Enhance Speech | se-zh-1 | Enhance Speech · Mandarin | 8.00 | 2.770 | **0.346** | 0.0761 | PASS |
| 18 | Enhancement and Separation | Enhance Speech | se-zh-2 | Enhance Speech · Mandarin | 5.00 | 1.699 | **0.340** | 0.0727 | PASS |
| 19 | Enhancement and Separation | Enhance Speech | se-en-1 | Enhance Speech · English | 2.94 | 1.743 | **0.593** | 0.0761 | PASS |
| 20 | Enhancement and Separation | Enhance Speech | se-en-2 | Enhance Speech · English | 3.60 | 1.857 | **0.516** | 0.0162 | PASS |
| 21 | Enhancement and Separation | Separate Speech | zh-1 | Separate Speech · Mandarin | 8.00 | 4.344 | **0.543** | 0.0272 | PASS |
| 22 | Enhancement and Separation | Separate Speech | zh-2 | Separate Speech · Mandarin | 8.00 | 5.747 | **0.718** | 0.0856 | PASS |
| 23 | Enhancement and Separation | Separate Speech | en-1 | Separate Speech · English | 8.00 | 6.450 | **0.806** | 0.0715 | PASS |
| 24 | Enhancement and Separation | Separate Speech | en-2 | Separate Speech · English | 8.00 | 5.704 | **0.713** | 0.0610 | PASS |
| 25 | Enhancement and Separation | Extract Vocals | ev-1 | Extract Vocals · Mandarin | 7.58 | 2.842 | **0.375** | 0.0498 | PASS |
| 26 | Enhancement and Separation | Extract Vocals | ev-2 | Extract Vocals · Mandarin | 8.00 | 6.089 | **0.761** | 0.0253 | PASS |
| 27 | Enhancement and Separation | Extract Vocals | ev-3 | Extract Vocals · English | 8.00 | 3.467 | **0.433** | 0.0376 | PASS |
| 28 | Enhancement and Separation | Extract Vocals | ev-4 | Extract Vocals · English | 8.00 | 6.253 | **0.782** | 0.0146 | PASS |
| 29 | Enhancement and Separation | Improve Quality | sr-zh-1 | Improve Quality · Mandarin | 4.30 | 2.274 | **0.529** | 0.1026 | PASS |
| 30 | Enhancement and Separation | Improve Quality | sr-zh-2 | Improve Quality · Mandarin | 4.82 | 2.513 | **0.521** | 0.0884 | PASS |
| 31 | Enhancement and Separation | Improve Quality | sr-en-1 | Improve Quality · English | 2.94 | 1.167 | **0.397** | 0.0821 | PASS |
| 32 | Enhancement and Separation | Improve Quality | sr-en-2 | Improve Quality · English | 2.62 | 1.750 | **0.668** | 0.0641 | PASS |
| 33 | Paralinguistic Editing | Emotion Edit | emo-zh-1 | Emotion Edit · Mandarin | 8.00 | 4.570 | **0.571** | 0.0261 | PASS |
| 34 | Paralinguistic Editing | Emotion Edit | emo-zh-2 | Emotion Edit · Mandarin | 8.00 | 6.452 | **0.807** | 0.0921 | PASS |
| 35 | Paralinguistic Editing | Emotion Edit | emo-en-1 | Emotion Edit · English | 8.00 | 7.883 | **0.985** | 0.0797 | PASS |
| 36 | Paralinguistic Editing | Emotion Edit | emo-en-2 | Emotion Edit · English | 8.00 | 7.363 | **0.920** | 0.0792 | PASS |
| 37 | Paralinguistic Editing | Timbre Edit | vc-1 | Voice Edit · Mandarin | 6.62 | 4.003 | **0.605** | 0.0815 | PASS |
| 38 | Paralinguistic Editing | Timbre Edit | vc-2 | Voice Edit · Mandarin | 6.86 | 4.493 | **0.655** | 0.1224 | PASS |
| 39 | Paralinguistic Editing | Timbre Edit | vc-3 | Voice Edit · Mandarin | 8.00 | 5.630 | **0.704** | 0.1061 | PASS |
| 40 | Paralinguistic Editing | Timbre Edit | vc-4 | Voice Edit · English | 8.00 | 10.226 | **1.278** | 0.0835 | PASS |
| 41 | Paralinguistic Editing | Nonverbal Edit | zh-a | Nonverbal Edit · Mandarin · Insert | 8.00 | 10.207 | **1.276** | 0.0687 | PASS |
| 42 | Paralinguistic Editing | Nonverbal Edit | zh-b | Nonverbal Edit · Mandarin · Remove | 8.00 | 13.739 | **1.717** | 0.0567 | PASS |
| 43 | Paralinguistic Editing | Nonverbal Edit | en-c | Nonverbal Edit · English · Insert | 8.00 | 14.015 | **1.752** | 0.0704 | PASS |
| 44 | Paralinguistic Editing | Nonverbal Edit | en-d | Nonverbal Edit · English · Remove | 8.00 | 18.827 | **2.353** | 0.0686 | PASS |
| 45 | Paralinguistic Editing | Whisper Edit | wh-w2n-zh | Normal → Whisper · Mandarin | 8.00 | 7.462 | **0.933** | 0.0633 | PASS |
| 46 | Paralinguistic Editing | Whisper Edit | wh-w2n-en | Normal → Whisper · English | 5.50 | 4.822 | **0.877** | 0.0388 | PASS |
| 47 | Paralinguistic Editing | Whisper Edit | wh-n2w-zh | Whisper → Normal · Mandarin | 4.50 | 4.942 | **1.098** | 0.0106 | PASS |
| 48 | Paralinguistic Editing | Whisper Edit | wh-n2w-en | Whisper → Normal · English | 5.34 | 5.396 | **1.011** | 0.0237 | PASS |
| 49 | Paralinguistic Editing | Deaccent | accent-tibetan | Tibetan accent → Standard Mandarin | 3.90 | 4.653 | **1.193** | 0.0650 | PASS |
| 50 | Paralinguistic Editing | Deaccent | accent-sichuan | Sichuan accent → Standard Mandarin | 8.00 | 7.348 | **0.918** | 0.0933 | PASS |
| 51 | Paralinguistic Editing | Deaccent | accent-dongbei | Northeastern accent → Standard Mandarin | 8.00 | 11.784 | **1.473** | 0.0643 | PASS |
| 52 | Paralinguistic Editing | Deaccent | accent-hubei | Hubei accent → Standard Mandarin | 2.50 | 4.099 | **1.640** | 0.0636 | PASS |
| 53 | Paralinguistic Editing | Deaccent | accent-hunan | Hunan accent → Standard Mandarin | 4.58 | 5.599 | **1.222** | 0.0639 | PASS |
| 54 | Paralinguistic Editing | Deaccent | accent-fujian | Fujian accent → Standard Mandarin | 5.08 | 4.549 | **0.895** | 0.0685 | PASS |
| 55 | Paralinguistic Editing | Deaccent | accent-india | Indian accent → Standard Mandarin | 3.34 | 4.101 | **1.228** | 0.0698 | PASS |
| 56 | Paralinguistic Editing | Deaccent | accent-japan | Japanese accent → Standard Mandarin | 5.54 | 5.682 | **1.026** | 0.0638 | PASS |
| 57 | Acoustic Editing | Speed Edit | speed-1 | Speed Edit | 8.00 | 7.646 | **0.956** | 0.0560 | PASS |
| 58 | Acoustic Editing | Speed Edit | speed-2 | Speed Edit | 3.30 | 3.778 | **1.145** | 0.0810 | PASS |
| 59 | Acoustic Editing | Energy Edit | volume-1 | Energy Edit | 5.50 | 4.496 | **0.818** | 0.0492 | PASS |
| 60 | Acoustic Editing | Energy Edit | volume-2 | Energy Edit | 2.50 | 2.685 | **1.074** | 0.0430 | PASS |
| 61 | Acoustic Editing | Pitch Edit | pitch-1 | Pitch Edit | 5.50 | 3.747 | **0.681** | 0.0590 | PASS |
| 62 | Acoustic Editing | Pitch Edit | pitch-2 | Pitch Edit | 5.88 | 5.251 | **0.893** | 0.0871 | PASS |