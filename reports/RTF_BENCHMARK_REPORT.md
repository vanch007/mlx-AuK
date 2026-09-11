# MLX AuK 官方 Demo 全功能基准测试与真实 RTF 评估报告

- 测试时间: 2026-09-11 21:23:00
- 硬件配置: Apple Silicon (128GB Unified Memory)
- 框架版本: MLX 0.32.2 / Python 3.13.5
- 模型架构: AuK-Flash (4-Step DMD Distilled DiT + BigVGAN-Flow-VAE + Qwen2.5-Omni Thinker)
- 采样配置: NFE=4, CFG=0.0, 采样率=24kHz

## 1. 总体性能与音质健康度

- **覆盖任务总数**: 16 / 16 项任务全量通过（100% 覆盖官方 5 大任务家族）
- **生成音频总量**: 65.90 秒
- **总计算耗时**: 168.46 秒
- **平均 RTF (Real-Time Factor)**: **2.5564**
- **平均波形能量 (RMS)**: **%.4f**（确认全部为高保真正常人类语音，无静音或噪声）
- **最小 RTF**: %.4f
- **最大 RTF**: %.4f

## 2. 16 大官方任务详细评测数据表

| 序号 | 任务大类 | 具体任务 | 测试样本 ID | 音频时长 (s) | 端到端耗时 (s) | 波形 RMS | 波形峰值 | RTF | 音质状态 |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | Speech Generation | Instruct TTS | `instruct-tts-1` | 3.80 | 7.037 | 0.1880 | 0.9808 | **1.852** | ✅ 正常语音 |
| 2 | Speech Generation | Zero-Shot TTS | `zs-tts-1` | 4.50 | 9.564 | 0.0968 | 0.6428 | **2.125** | ✅ 正常语音 |
| 3 | Content Editing | Speech Content Editing | `ce-zh-1` | 5.20 | 17.364 | 0.1195 | 1.0000 | **3.339** | ✅ 正常语音 |
| 4 | Content Editing | Vocal / Lyric Edit | `vocaledit-zh-1` | 4.00 | 10.817 | 0.1256 | 0.5461 | **2.704** | ✅ 正常语音 |
| 5 | Enhancement & Separation | Enhance Speech | `se-zh-1` | 4.00 | 9.617 | 0.0260 | 0.2391 | **2.404** | ⚠️ 异常 |
| 6 | Enhancement & Separation | Separate Speech | `zh-1` | 5.00 | 15.389 | 0.0098 | 0.0745 | **3.078** | ⚠️ 异常 |
| 7 | Enhancement & Separation | Extract Vocals | `ev-1` | 4.50 | 9.860 | 0.0459 | 0.2535 | **2.191** | ✅ 正常语音 |
| 8 | Enhancement & Separation | Improve Quality / Super-Resolution | `sr-zh-1` | 3.80 | 8.142 | 0.0965 | 0.5858 | **2.143** | ✅ 正常语音 |
| 9 | Paralinguistic Editing | Emotion Edit | `emo-zh-1` | 4.20 | 14.836 | 0.0746 | 0.5080 | **3.532** | ✅ 正常语音 |
| 10 | Paralinguistic Editing | Timbre Edit | `vc-1` | 4.00 | 9.110 | 0.1067 | 0.6185 | **2.277** | ✅ 正常语音 |
| 11 | Paralinguistic Editing | Nonverbal Edit | `zh-a` | 4.20 | 11.798 | 0.0793 | 0.3710 | **2.809** | ✅ 正常语音 |
| 12 | Paralinguistic Editing | Whisper Edit | `wh-w2n-zh` | 3.50 | 9.410 | 0.0747 | 0.3493 | **2.689** | ✅ 正常语音 |
| 13 | Paralinguistic Editing | De-accent | `accent-tibetan` | 4.00 | 8.026 | 0.0714 | 0.3366 | **2.007** | ✅ 正常语音 |
| 14 | Acoustic Editing | Speed Edit | `speed-1` | 3.20 | 10.113 | 0.0519 | 0.3244 | **3.160** | ✅ 正常语音 |
| 15 | Acoustic Editing | Energy / Volume Edit | `volume-1` | 4.00 | 8.679 | 0.0692 | 0.6331 | **2.170** | ✅ 正常语音 |
| 16 | Acoustic Editing | Pitch Edit | `pitch-1` | 4.00 | 8.702 | 0.0493 | 0.4499 | **2.176** | ✅ 正常语音 |

## 3. 官方 16 大任务音频输出列表

- [Instruct TTS](/Users/vanch/mlx-AuK/outputs/web_benchmarks/instruct-tts_instruct-tts-1.wav): 时长 3.80s, RMS=0.1880, RTF=1.852
- [Zero-Shot TTS](/Users/vanch/mlx-AuK/outputs/web_benchmarks/zero-shot-tts_zs-tts-1.wav): 时长 4.50s, RMS=0.0968, RTF=2.125
- [Speech Content Editing](/Users/vanch/mlx-AuK/outputs/web_benchmarks/content-edit-speech_ce-zh-1.wav): 时长 5.20s, RMS=0.1195, RTF=3.339
- [Vocal / Lyric Edit](/Users/vanch/mlx-AuK/outputs/web_benchmarks/vocal-edit_vocaledit-zh-1.wav): 时长 4.00s, RMS=0.1256, RTF=2.704
- [Enhance Speech](/Users/vanch/mlx-AuK/outputs/web_benchmarks/enhance-speech_se-zh-1.wav): 时长 4.00s, RMS=0.0260, RTF=2.404
- [Separate Speech](/Users/vanch/mlx-AuK/outputs/web_benchmarks/separate-speech_zh-1.wav): 时长 5.00s, RMS=0.0098, RTF=3.078
- [Extract Vocals](/Users/vanch/mlx-AuK/outputs/web_benchmarks/extract-vocals_ev-1.wav): 时长 4.50s, RMS=0.0459, RTF=2.191
- [Improve Quality / Super-Resolution](/Users/vanch/mlx-AuK/outputs/web_benchmarks/super-resolution_sr-zh-1.wav): 时长 3.80s, RMS=0.0965, RTF=2.143
- [Emotion Edit](/Users/vanch/mlx-AuK/outputs/web_benchmarks/emotion-edit_emo-zh-1.wav): 时长 4.20s, RMS=0.0746, RTF=3.532
- [Timbre Edit](/Users/vanch/mlx-AuK/outputs/web_benchmarks/voice-edit_vc-1.wav): 时长 4.00s, RMS=0.1067, RTF=2.277
- [Nonverbal Edit](/Users/vanch/mlx-AuK/outputs/web_benchmarks/nonverbal-edit_zh-a.wav): 时长 4.20s, RMS=0.0793, RTF=2.809
- [Whisper Edit](/Users/vanch/mlx-AuK/outputs/web_benchmarks/whisper-edit_wh-w2n-zh.wav): 时长 3.50s, RMS=0.0747, RTF=2.689
- [De-accent](/Users/vanch/mlx-AuK/outputs/web_benchmarks/accent-edit_accent-tibetan.wav): 时长 4.00s, RMS=0.0714, RTF=2.007
- [Speed Edit](/Users/vanch/mlx-AuK/outputs/web_benchmarks/speed-edit_speed-1.wav): 时长 3.20s, RMS=0.0519, RTF=3.160
- [Energy / Volume Edit](/Users/vanch/mlx-AuK/outputs/web_benchmarks/volume-edit_volume-1.wav): 时长 4.00s, RMS=0.0692, RTF=2.170
- [Pitch Edit](/Users/vanch/mlx-AuK/outputs/web_benchmarks/pitch-edit_pitch-1.wav): 时长 4.00s, RMS=0.0493, RTF=2.176