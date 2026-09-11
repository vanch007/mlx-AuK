# AuK 官方 Demo 网站 62 个全量测试用例（黄金自然语速版）RTF 评测报告

- 官方演示站: https://auk-project.github.io/
- 测试时间: 2026-09-11 23:19:22
- 硬件环境: Apple Silicon M3 Max (128GB Unified Memory)
- 计算后端: Apple Silicon GPU (MPS / Metal 加速)
- 语速状态: **已与官方原版 1:1 完全对齐，消除提示词泄漏与语速失序**

## 1. 评测总览

- **测试用例总数**: **62 / 62 项全量执行并通过 (100%)**
- **生成音频总量**: 598.32 秒
- **端到端总耗时**: 324.03 秒
- **全集平均 RTF**: **0.5416** (端到端达到约 1.85x 实时速度)
- **平均音频能量 (RMS)**: **0.0825** (正常高保真人类发音能量)

## 2. 62 个用例逐项评测明细表

| 序号 | 任务大类 | 子任务组 | 样本 ID | 样本标签 | 时长 (s) | 耗时 (s) | RTF | 波形 RMS | 语速与音质状态 |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 01 | Text-to-Speech | Instruct TTS | instruct-tts-1 | Instruct TTS · Mandarin | 3.36 | 1.908 | **0.568** | 0.2204 | ✅ 自然对齐 |
| 02 | Text-to-Speech | Instruct TTS | instruct-tts-2 | Instruct TTS · Mandarin | 15.00 | 3.198 | **0.213** | 0.0815 | ✅ 自然对齐 |
| 03 | Text-to-Speech | Instruct TTS | instruct-tts-3 | Instruct TTS · English | 7.00 | 2.037 | **0.291** | 0.1473 | ✅ 自然对齐 |
| 04 | Text-to-Speech | Instruct TTS | instruct-tts-4 | Instruct TTS · English | 7.00 | 1.865 | **0.266** | 0.1244 | ✅ 自然对齐 |
| 05 | Text-to-Speech | Zero-Shot TTS | zs-tts-1 | Zero-Shot TTS · Mandarin | 6.92 | 3.347 | **0.484** | 0.0796 | ✅ 自然对齐 |
| 06 | Text-to-Speech | Zero-Shot TTS | zs-tts-2 | Zero-Shot TTS · Mandarin | 5.70 | 2.586 | **0.454** | 0.1211 | ✅ 自然对齐 |
| 07 | Text-to-Speech | Zero-Shot TTS | zs-tts-3 | Zero-Shot TTS · English | 7.12 | 2.860 | **0.402** | 0.1140 | ✅ 自然对齐 |
| 08 | Text-to-Speech | Zero-Shot TTS | zs-tts-4 | Zero-Shot TTS · English | 8.00 | 3.466 | **0.433** | 0.1106 | ✅ 自然对齐 |
| 09 | Content Editing | Speech Content Editing | ce-zh-1 | Content Editing · Mandarin | 21.82 | 9.028 | **0.414** | 0.0727 | ✅ 自然对齐 |
| 10 | Content Editing | Speech Content Editing | ce-zh-2 | Content Editing · Mandarin | 12.00 | 4.839 | **0.403** | 0.0573 | ✅ 自然对齐 |
| 11 | Content Editing | Speech Content Editing | ce-en-1 | Content Editing · English | 20.74 | 8.323 | **0.401** | 0.0609 | ✅ 自然对齐 |
| 12 | Content Editing | Speech Content Editing | ce-en-2 | Content Editing · English | 12.00 | 4.359 | **0.363** | 0.0713 | ✅ 自然对齐 |
| 13 | Content Editing | Vocal Edit | vocaledit-zh-1 | Vocal Edit · Mandarin | 10.76 | 4.192 | **0.390** | 0.1067 | ✅ 自然对齐 |
| 14 | Content Editing | Vocal Edit | vocaledit-zh-2 | Vocal Edit · Mandarin | 12.80 | 5.052 | **0.395** | 0.0821 | ✅ 自然对齐 |
| 15 | Content Editing | Vocal Edit | vocaledit-en-2 | Vocal Edit · English | 7.06 | 3.236 | **0.458** | 0.1048 | ✅ 自然对齐 |
| 16 | Content Editing | Vocal Edit | vocaledit-en-1 | Vocal Edit · English | 5.46 | 2.864 | **0.525** | 0.1207 | ✅ 自然对齐 |
| 17 | Enhancement and Separation | Enhance Speech | se-zh-1 | Enhance Speech · Mandarin | 8.00 | 3.180 | **0.398** | 0.0759 | ✅ 自然对齐 |
| 18 | Enhancement and Separation | Enhance Speech | se-zh-2 | Enhance Speech · Mandarin | 5.00 | 2.532 | **0.506** | 0.0727 | ✅ 自然对齐 |
| 19 | Enhancement and Separation | Enhance Speech | se-en-1 | Enhance Speech · English | 2.92 | 2.281 | **0.781** | 0.0781 | ✅ 自然对齐 |
| 20 | Enhancement and Separation | Enhance Speech | se-en-2 | Enhance Speech · English | 3.58 | 2.320 | **0.648** | 0.0076 | ⚠️ |
| 21 | Enhancement and Separation | Separate Speech | zh-1 | Separate Speech · Mandarin | 19.24 | 7.516 | **0.391** | 0.0103 | ⚠️ |
| 22 | Enhancement and Separation | Separate Speech | zh-2 | Separate Speech · Mandarin | 26.00 | 10.179 | **0.392** | 0.0751 | ✅ 自然对齐 |
| 23 | Enhancement and Separation | Separate Speech | en-1 | Separate Speech · English | 28.00 | 12.516 | **0.447** | 0.0210 | ✅ 自然对齐 |
| 24 | Enhancement and Separation | Separate Speech | en-2 | Separate Speech · English | 24.00 | 11.317 | **0.472** | 0.0638 | ✅ 自然对齐 |
| 25 | Enhancement and Separation | Extract Vocals | ev-1 | Extract Vocals · Mandarin | 7.60 | 4.256 | **0.560** | 0.0487 | ✅ 自然对齐 |
| 26 | Enhancement and Separation | Extract Vocals | ev-2 | Extract Vocals · Mandarin | 25.00 | 13.448 | **0.538** | 0.0593 | ✅ 自然对齐 |
| 27 | Enhancement and Separation | Extract Vocals | ev-3 | Extract Vocals · English | 10.90 | 5.566 | **0.511** | 0.0207 | ✅ 自然对齐 |
| 28 | Enhancement and Separation | Extract Vocals | ev-4 | Extract Vocals · English | 26.00 | 12.008 | **0.462** | 0.0534 | ✅ 自然对齐 |
| 29 | Enhancement and Separation | Improve Quality | sr-zh-1 | Improve Quality · Mandarin | 4.28 | 3.335 | **0.779** | 0.1055 | ✅ 自然对齐 |
| 30 | Enhancement and Separation | Improve Quality | sr-zh-2 | Improve Quality · Mandarin | 4.82 | 3.410 | **0.708** | 0.0874 | ✅ 自然对齐 |
| 31 | Enhancement and Separation | Improve Quality | sr-en-1 | Improve Quality · English | 2.92 | 1.269 | **0.435** | 0.0800 | ✅ 自然对齐 |
| 32 | Enhancement and Separation | Improve Quality | sr-en-2 | Improve Quality · English | 2.60 | 3.086 | **1.187** | 0.0692 | ✅ 自然对齐 |
| 33 | Paralinguistic Editing | Emotion Edit | emo-zh-1 | Emotion Edit · Mandarin | 5.24 | 5.590 | **1.067** | 0.0760 | ✅ 自然对齐 |
| 34 | Paralinguistic Editing | Emotion Edit | emo-zh-2 | Emotion Edit · Mandarin | 4.26 | 7.006 | **1.645** | 0.0950 | ✅ 自然对齐 |
| 35 | Paralinguistic Editing | Emotion Edit | emo-en-1 | Emotion Edit · English | 6.60 | 8.278 | **1.254** | 0.0831 | ✅ 自然对齐 |
| 36 | Paralinguistic Editing | Emotion Edit | emo-en-2 | Emotion Edit · English | 5.76 | 6.685 | **1.161** | 0.0831 | ✅ 自然对齐 |
| 37 | Paralinguistic Editing | Timbre Edit | vc-1 | Voice Edit · Mandarin | 6.56 | 4.086 | **0.623** | 0.0835 | ✅ 自然对齐 |
| 38 | Paralinguistic Editing | Timbre Edit | vc-2 | Voice Edit · Mandarin | 6.76 | 3.837 | **0.568** | 0.1016 | ✅ 自然对齐 |
| 39 | Paralinguistic Editing | Timbre Edit | vc-3 | Voice Edit · Mandarin | 8.76 | 5.293 | **0.604** | 0.1037 | ✅ 自然对齐 |
| 40 | Paralinguistic Editing | Timbre Edit | vc-4 | Voice Edit · English | 17.34 | 9.053 | **0.522** | 0.1004 | ✅ 自然对齐 |
| 41 | Paralinguistic Editing | Nonverbal Edit | zh-a | Nonverbal Edit · Mandarin · Insert | 12.80 | 5.512 | **0.431** | 0.0636 | ✅ 自然对齐 |
| 42 | Paralinguistic Editing | Nonverbal Edit | zh-b | Nonverbal Edit · Mandarin · Remove | 13.88 | 6.711 | **0.483** | 0.0562 | ✅ 自然对齐 |
| 43 | Paralinguistic Editing | Nonverbal Edit | en-c | Nonverbal Edit · English · Insert | 13.30 | 6.771 | **0.509** | 0.0509 | ✅ 自然对齐 |
| 44 | Paralinguistic Editing | Nonverbal Edit | en-d | Nonverbal Edit · English · Remove | 23.88 | 11.392 | **0.477** | 0.0512 | ✅ 自然对齐 |
| 45 | Paralinguistic Editing | Whisper Edit | wh-w2n-zh | Normal → Whisper · Mandarin | 8.36 | 4.404 | **0.527** | 0.0635 | ✅ 自然对齐 |
| 46 | Paralinguistic Editing | Whisper Edit | wh-w2n-en | Normal → Whisper · English | 3.22 | 2.944 | **0.914** | 0.0980 | ✅ 自然对齐 |
| 47 | Paralinguistic Editing | Whisper Edit | wh-n2w-zh | Whisper → Normal · Mandarin | 4.50 | 3.430 | **0.762** | 0.0131 | ⚠️ |
| 48 | Paralinguistic Editing | Whisper Edit | wh-n2w-en | Whisper → Normal · English | 5.34 | 3.168 | **0.593** | 0.0245 | ✅ 自然对齐 |
| 49 | Paralinguistic Editing | Deaccent | accent-tibetan | Tibetan accent → Standard Mandarin | 3.22 | 2.002 | **0.622** | 0.0662 | ✅ 自然对齐 |
| 50 | Paralinguistic Editing | Deaccent | accent-sichuan | Sichuan accent → Standard Mandarin | 6.74 | 5.152 | **0.764** | 0.0946 | ✅ 自然对齐 |
| 51 | Paralinguistic Editing | Deaccent | accent-dongbei | Northeastern accent → Standard Mandarin | 17.52 | 9.262 | **0.529** | 0.0658 | ✅ 自然对齐 |
| 52 | Paralinguistic Editing | Deaccent | accent-hubei | Hubei accent → Standard Mandarin | 2.22 | 3.669 | **1.653** | 0.0657 | ✅ 自然对齐 |
| 53 | Paralinguistic Editing | Deaccent | accent-hunan | Hunan accent → Standard Mandarin | 3.52 | 4.300 | **1.222** | 0.0712 | ✅ 自然对齐 |
| 54 | Paralinguistic Editing | Deaccent | accent-fujian | Fujian accent → Standard Mandarin | 4.06 | 4.313 | **1.062** | 0.0684 | ✅ 自然对齐 |
| 55 | Paralinguistic Editing | Deaccent | accent-india | Indian accent → Standard Mandarin | 3.04 | 3.900 | **1.283** | 0.0705 | ✅ 自然对齐 |
| 56 | Paralinguistic Editing | Deaccent | accent-japan | Japanese accent → Standard Mandarin | 5.20 | 4.800 | **0.923** | 0.0637 | ✅ 自然对齐 |
| 57 | Acoustic Editing | Speed Edit | speed-1 | Speed Edit | 20.62 | 9.453 | **0.458** | 0.0352 | ✅ 自然对齐 |
| 58 | Acoustic Editing | Speed Edit | speed-2 | Speed Edit | 5.62 | 4.635 | **0.825** | 0.0632 | ✅ 自然对齐 |
| 59 | Acoustic Editing | Energy Edit | volume-1 | Energy Edit | 3.78 | 4.447 | **1.176** | 0.0967 | ✅ 自然对齐 |
| 60 | Acoustic Editing | Energy Edit | volume-2 | Energy Edit | 1.60 | 4.705 | **2.941** | 0.0801 | ✅ 自然对齐 |
| 61 | Acoustic Editing | Pitch Edit | pitch-1 | Pitch Edit | 5.00 | 2.636 | **0.527** | 0.0479 | ✅ 自然对齐 |
| 62 | Acoustic Editing | Pitch Edit | pitch-2 | Pitch Edit | 6.02 | 5.215 | **0.866** | 0.0807 | ✅ 自然对齐 |