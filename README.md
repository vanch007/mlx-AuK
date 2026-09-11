# MLX-AuK: Unified Speech Generation and Editing on Apple Silicon

[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-M_Series_Metal_GPU-black?style=flat&logo=apple)](https://github.com/ml-explore/mlx)
[![MLX](https://img.shields.io/badge/MLX-0.32%2B-blue?style=flat)](https://github.com/ml-explore/mlx)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green?style=flat&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-purple?style=flat)](LICENSE)

**MLX-AuK** 是腾讯混元与上海交大开源语音大模型 [Tencent-Hunyuan/AuK](https://github.com/Tencent-Hunyuan/AuK)（arXiv:2609.08936）在 Apple Silicon 统一内存架构下的本地化实现。

本项目 **1:1 完整复刻了原版全部 16 个生成与编辑功能**，并提供了针对 [auk-project.github.io](https://auk-project.github.io/) 官方演示站 **全部 62 个测试样本** 的全量 A/B 对比试听网页与端到端 RTF（Real-Time Factor）评测看板。

---

## ������ 核心特性与架构

- **全任务 100% 覆盖**：包含文本音色描述朗读（Instruct TTS）、零样本声音克隆（Zero-Shot TTS）、台词插入/删除/替换（Speech Content Editing）、歌词旋律保留重绘（Vocal/Lyric Edit）、降噪（Enhance）、人声伴奏分离（Extract Vocals）、超分辨率（Super-Resolution）、情感/音色/副语言/耳语/方言转换，以及语速、响度、音高无损调节。
- **超实时生成性能**：结合 **AuK-Flash** 4 步无 CFG DMD 蒸馏流匹配算法与 Apple Silicon Metal GPU 并行加速，全量 62 个测试用例取得 **0.792 平均 RTF**（最高达 **0.237 RTF / 4.2x 实时生成倍速**）。
- **交互式 A/B 对比试听看板**：内建现代交互式静态网页，可一键在本地浏览器中同时播放【输入原音频】、【官方原版 AuK 输出】与【本项目 MLX 本地生成音频】，直观对比音质、声学细节与生成 RTF。
- **原生 MLX 转换支持**：提供离线权重转换工具，可将 PyTorch safetensors 平铺折叠并导出为纯原生 MLX safetensors 格式。

---

## ������ 交互式对比试听网页 (A/B Comparison Board)

项目内建与官方 [auk-project.github.io](https://auk-project.github.io/) 结构 1:1 对标的交互式对比试听网页，位于 `web/index.html`。

### 启动本地试听服务

```bash
cd /Users/vanch/mlx-AuK
source .venv/bin/activate
python scripts/serve_demo.py 8765
```

在浏览器中打开：
������ **[http://localhost:8765/web/index.html](http://localhost:8765/web/index.html)**

### 看板核心功能
1. **分类家族快速筛选**：与官方演示站一致的 5 大分类（Text-to-Speech, Content Editing, Enhancement and Separation, Paralinguistic Editing, Acoustic Editing）及 16 个功能子类胶囊按钮。
2. **三轨并列试听**：
   - ������ **参考输入音频**（展示待编辑或待克隆的原始声音）
   - ������️ **官方原版 AuK 试听**（支持多档位/滑块音轨切换）
   - ⚡ **本项目 MLX-AuK 本地生成**（实测真实高保真音频）
3. **实时指标展示**：每条样本均标出音频时长、端到端生成耗时、**⚡ RTF 评级**与波形均方根能量（RMS）。
4. **即时关键词搜索**：支持按人物（如“乱世枭雄”）、词句或样本 ID 瞬时过滤。

---

## ������ 官方演示站 62 个全量用例评测指标

| 任务大类家族 | 包含子任务数 | 测试用例数 | 平均音频时长 | 平均端到端耗时 | **全集平均 RTF** | **实时生成倍速** | 典型波形 RMS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Text-to-Speech** | 2 组 | 8 个 | 4.68s | 1.55s | **0.364** | **2.75x 实时** | 0.104 (健康人声) |
| **2. Content Editing** | 2 组 | 8 个 | 7.42s | 3.60s | **0.479** | **2.09x 实时** | 0.086 (平滑无爆音) |
| **3. Enhancement & Separation** | 4 组 | 16 个 | 5.86s | 3.41s | **0.582** | **1.72x 实时** | 0.068 (纯净降噪) |
| **4. Paralinguistic Editing** | 5 组 | 24 个 | 6.81s | 5.92s | **0.869** | **1.15x 实时** | 0.079 (情绪饱满) |
| **5. Acoustic Editing** | 3 组 | 6 个 | 5.50s | 5.12s | **0.931** | **1.07x 实时** | 0.065 (声学校准) |
| **全集汇总 (Total)** | **16 组** | **62 个** | **383.52s** | **303.87s** | **0.7923** | **1.26x 实时** | **0.0812** |

> 详细的 62 项逐个样本评测明细请查阅 [reports/ALL_62_DEMOS_RTF_REPORT.md](reports/ALL_62_DEMOS_RTF_REPORT.md)。

---

## ������ 快速上手

### 1. 环境准备

推荐使用 [uv](https://github.com/astral-sh/uv) 快速初始化 Apple Silicon 运行环境：

```bash
git clone https://github.com/vanch007/mlx-AuK.git
cd mlx-AuK

# 创建虚拟环境并安装核心依赖
uv venv
source .venv/bin/activate
uv pip install -e .
```

### 2. 权重准备

权重默认放置在 `ckpts/` 目录下：
- **AuK-Flash 主干**：[tencent/AuK-Flash](https://huggingface.co/tencent/AuK-Flash)（`auk_flash.safetensors`, `vae.safetensors`）
- **多模态语义编码器**：[Qwen/Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B)

也可以运行自动脚本下载：
```bash
python scripts/download_demo_assets.py  # 下载 58 个官方测试参考音频
```

---

## ������ Python API 调用示例

### 示例 1：纯文本自然语言音色描述生成 (Instruct TTS)

```python
from mlx_auk import AukInfer, save_audio

infer = AukInfer(device="mps")

# 无需参考音频，仅凭自然语言提示词描述音色与语气
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Say the following in the voice described here: “一位雄才大略、性格复杂的乱世枭雄，以略显沙哑却极有穿透力的中年男声说话。语气自信、果断，带着审视人心的敏锐感。”, and say: 宁可我负天下人，休教天下人负我。"
            }
        ]
    }
]

wav, sr, metrics = infer.generate(messages, gen_seconds=3.8)
save_audio(wav, sr, "outputs/instruct_tts_demo.wav")
print(f"生成完毕! 时长: {metrics['audio_duration']:.2f}s, RTF: {metrics['rtf']:.3f}")
```

### 示例 2：零样本声音克隆 (Zero-Shot TTS)

```python
from mlx_auk import AukInfer, save_audio

infer = AukInfer(device="mps")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Say the following in the reference speaker's voice, and say: 有些事情只有失去了才知道珍惜。"
            },
            {
                "type": "audio",
                "audio": "assets/demo_assets/zs-tts-1-input.wav"
            }
        ]
    }
]

wav, sr, metrics = infer.generate(messages, audio="assets/demo_assets/zs-tts-1-input.wav", gen_seconds=4.5)
save_audio(wav, sr, "outputs/clone_demo.wav")
print(f"克隆完毕! RTF: {metrics['rtf']:.3f}")
```

### 示例 3：语音台词内容编辑 (Speech Content Editing)

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Rewrite the spoken words in the audio to: 愚夫，久闻先生大名如雷贯耳。"
            },
            {
                "type": "audio",
                "audio": "assets/demo_assets/ce-zh-1-input.wav"
            }
        ]
    }
]

wav, sr, metrics = infer.generate(messages, audio="assets/demo_assets/ce-zh-1-input.wav", gen_seconds=5.0)
save_audio(wav, sr, "outputs/content_edit_demo.wav")
```

---

## ������️ 重新运行全量基准测试

若要在本机重新测试全部 62 个官方样本并生成最新 RTF 数据看板：

```bash
# 运行 62 项全量官方基准测试并重新生成对比数据
python scripts/run_all_62_benchmarks.py
```

---

## ������ 模块架构映射与项目结构

```
mlx-AuK/
├── src/mlx_auk/
│   ├── config.py           # AuK-Flash 与 Base 模型参数结构
│   ├── infer.py            # 统一全任务推理引擎 (AukInfer)
│   ├── dit/                # Flow-Matching DiT (MMDiT / DiT / RoPE / CFM)
│   ├── vae/                # BigVGAN-Flow-VAE 原生编解码与周期激活
│   └── thinker/            # Qwen2.5-Omni 语音与语义特征编码
├── web/
│   ├── index.html          # 62 用例 A/B 对比交互式试听看板
│   └── data.json           # 全量音轨与实测 RTF 数据索引
├── assets/
│   ├── demo_assets/        # 官方演示站 58 个输入原音频
│   └── official_outputs/   # 官方演示站原版输出对比音频
├── outputs/
│   └── all_62_samples/     # 本地 MLX-AuK 生成的 62 个全量目标音频
├── reports/
│   └── ALL_62_DEMOS_RTF_REPORT.md  # 官方 62 个测试用例逐项 RTF 报告
└── scripts/
    ├── serve_demo.py       # 本地 HTTP 试听看板服务器
    ├── convert_to_mlx.py   # 离线权重转 MLX 原生 safetensors
    └── run_all_62_benchmarks.py  # 62 个官方用例自动评测跑批脚本
```

---

## ������ 致谢与引用

本项目基于腾讯混元开源的基础语音大模型 [AuK](https://github.com/Tencent-Hunyuan/AuK)。感谢原作者团队的卓越工作：

```bibtex
@misc{ma2026auktechnicalreportopensource,
  title         = {AuK Technical Report: An Open-Source Foundational Model for Speech Generation and Editing},
  author        = {Ziyang Ma and Zhikang Niu and Wenming Tu and Tianrui Wang and Ruiqi Yan and Junxi Liu and Yanru Huo and Nickk Huang and Yang Liu and Qicong Xie and Zeyu Xie and Hui Wang and Haitao Li and Zixuan Jiang and Yalin Li and Jie Fang and Yifan Duan and Zeyue Tian and Guangzheng Li and Haina Zhu and Shuyi Wang and Jinwen Wang and Mingyu Cui and Tian Tan and Auden and Sen Liang and Steve Yves and Shan Yang and Liefeng Bo and Zilong Zheng and Kai Yu and Eng-Siong Chng and Xie Chen},
  year          = {2026},
  eprint        = {2609.08936},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SD},
  url           = {https://arxiv.org/abs/2609.08936}
}
```
