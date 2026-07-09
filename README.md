# 无线数字孪生信道生成 · Physical-AI Framework

面向华为 **「基于 Physical AI 的无线数字孪生信道生成」** 赛题的解耦重构框架。
目标:给定环境地图 `M` 与位置坐标 `x`，用 Physical-AI 神经网络生成 MIMO-OFDM
信道 `H = f(M, x)`，并按 PAS / PDP / NMSE 指标评估。

> 本分支把原始 WRF-GS+ (3D-GS) 代码重构为 **数据 / 模型 / 训练 / 评估** 四层解耦
> 结构，并自带一个 **纯 PyTorch、免编译 CUDA** 的基线模型，**Linux 与 Windows
> 双系统开箱即用**。原 WRF-GS+ 说明见 [`docs/WRFGS_reference.md`](docs/WRFGS_reference.md)，
> 其 3D-GS 模型可作为可选后端接入（见下文）。

---

## 目录结构（解耦设计）

```
wireless_twin/                # 核心库：四层互不依赖具体实现
├── data/                     # ① 数据层：只负责读比赛文件
│   ├── setup_config.py       #    解析 RoundX_Setup.json -> ChannelSpec
│   ├── channel_dataset.py    #    Pos/Channel -> Dataset，含 load_round()
│   ├── normalization.py      #    信道归一化（可逆）
│   └── map_loader.py         #    读取 RoundX_Map.ply 点云
├── models/                   # ② 模型层：位置 -> 复信道，统一接口 ChannelModel
│   ├── base.py               #    抽象接口 forward(pos)->(B,M,N,S) complex
│   ├── path_field.py         #    ★ 基线：Fourier-MLP + CP 分解（跨平台）
│   ├── encodings.py          #    Fourier 位置编码
│   ├── registry.py           #    按名字构建模型
│   └── wrfgs_backend.py      #    可选 3D-GS(WRF-GS+) 后端接入点
├── training/                 # ③ 训练层：与模型/数据实现无关
│   ├── losses.py             #    NMSE + PAS/PDP 一致性损失（对齐排名指标）
│   └── trainer.py            #    Trainer.fit() / save_checkpoint()
├── evaluation/               # ④ 评估层
│   ├── metrics.py            #    C1(PAS) C2(PDP) C3(NMSE) 与综合分 C
│   └── predictor.py          #    生成 RoundX_Test_Channel.npy
└── signal.py                 #    PAS/PDP 变换（训练与评估共用，保证一致）

configs/round1.yaml           # 配置（模型/训练超参）
scripts/                      # 命令行入口
├── train.py                  #    训练
├── infer.py                  #    生成提交文件 RoundX_Test_Channel.npy
├── evaluate.py               #    本地离线评分
└── make_synthetic_data.py    #    生成小规模合成数据（跑通/自测用）
tests/test_pipeline.py        # 端到端冒烟测试
```

**解耦要点**：训练层只通过 `model(pos) -> complex H` 调用模型，不关心它是 MLP 还是
3D-GS；也只消费 `(位置, 目标)` 批次，不关心文件格式。因此四层可各自独立替换、演进。

---

## 环境安装

需要 Python ≥ 3.9。

### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-twin.txt
```

### Windows（PowerShell）
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-twin.txt
```

如需 CUDA 版 torch，请先按 <https://pytorch.org/get-started/locally/> 单独安装
`torch`，再装其余依赖。基线模型 CPU 也能训练（小规模）。

---

## 快速开始（用合成数据跑通全流程）

以下命令在 Linux 与 Windows 一致（Windows 下把 `python` 换成 `py` 亦可）：

```bash
# 0) 端到端自测（约几秒，验证四层已正确接线）
python tests/test_pipeline.py

# 1) 生成一份小的合成数据（格式与官方 RoundX_* 完全一致）
python scripts/make_synthetic_data.py --out data/DataSynth --round Round1

# 2) 训练
python scripts/train.py --config configs/round1.yaml \
    --datadir data/DataSynth --ckpt checkpoints/synth.pt --set train.epochs=50

# 3) 生成提交文件 Round1_Test_Channel.npy
python scripts/infer.py --ckpt checkpoints/synth.pt --datadir data/DataSynth

# 4) 本地评分（合成数据带 *_Test_Channel_GT.npy）
python scripts/evaluate.py \
    --pred data/DataSynth/Round1_Test_Channel.npy \
    --gt   data/DataSynth/Round1_Test_Channel_GT.npy \
    --setup data/DataSynth/Round1_Setup.json
```

## 用官方数据

把官方 `Data1` 解压到 `data/Data1/`（含 `Round1_Setup.json`、`Round1_Map.ply`、
`Round1_Train_Pos.npy`、`Round1_Train_Channel.npy`、`Round1_Test_Pos.npy`），然后：

```bash
python scripts/train.py --config configs/round1.yaml --datadir data/Data1 \
    --ckpt checkpoints/round1.pt
python scripts/infer.py --ckpt checkpoints/round1.pt --datadir data/Data1
# -> data/Data1/Round1_Test_Channel.npy  （即提交文件，复数 P×M×N×S）
```

---

## 评价指标（严格对应任务书 §2.2）

| 指标 | 含义 | 实现 |
|------|------|------|
| `C1` | PAS 功率角度谱余弦相似度 | `evaluation/metrics.py::pas_accuracy` |
| `C2` | PDP 功率时延谱余弦相似度 | `evaluation/metrics.py::pdp_accuracy` |
| `C3` | 信道 NMSE | `evaluation/metrics.py::channel_nmse` |
| `C`  | `w1·C1 + w2·C2 + w3·1/(1+C3)` | `competition_score` |

PAS/PDP 的谱变换在 [`wireless_twin/signal.py`](wireless_twin/signal.py) **统一定义**，
训练损失与评估共用同一实现。若官方评测的角度/时延约定与此不同，只需改这一个文件，
所有指标与损失自动跟随。训练损失 `L = NMSE + λ_pas·(1−C1) + λ_pdp·(1−C2)` 直接对齐
排名指标。

---

## 基线模型：PathField

把信道建模为 `K` 条传播「模式」的低秩(CP)叠加：

```
H(x)[m,n,s] = Σ_k  c_k(x) · U[k,m] · V[k,n] · W[k,s]
```

- `c_k(x)`：位置经 Fourier 编码 + MLP 得到的复增益；
- `U,V,W`：分别对应 BS 阵列 / UE 阵列 / 子载波(时延) 的可学习复签名。

参数量约 `K·(M+N+S)`，远小于逐元素输出，纯 PyTorch 实现，CPU/GPU/Windows 均可训练，
物理可解释。它是一个诚实的强基线，后续可在同一接口下换成 3D-GS。

## 可选：接入 WRF-GS+ (3D-GS) 后端

`models/wrfgs_backend.py` 是接入点（需 CUDA 光栅化子模块）。本精简仓库不包含 3D-GS
实现，原 WRF-GS+ 代码保留在本地 `main` 分支历史里，可用
`git checkout main -- gaussian_renderer scene arguments utils submodules` 取回，
再按 `wrfgs_backend.py` 的注释接线。只要实现同样的 `forward(pos)->complex (B,M,N,S)`
接口并 `register_model("wrfgs")`，即可在配置里切换 `model.name: wrfgs`，训练/评估代码
零改动。在未编译扩展的机器（如 Windows）上，该后端自动跳过，不影响基线运行。

---

## 备注

- 仅使用 AI 方法（禁止传统射线追踪），符合赛题要求。
- 提交格式：`RoundX_Test_Channel.npy`，复数张量 `P_Test × M × N × S`。
- 大文件（`data/`、`checkpoints/`、`*.npy`、`*.pt`）已在 `.gitignore` 中忽略。
