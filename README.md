<div align="center">

# 🐴 ↔ 🦓 &nbsp; CycleGAN — PyTorch Implementation

> **Unpaired image-to-image translation using cycle-consistent adversarial networks**  
> A clean, from-scratch PyTorch re-implementation of the seminal [CycleGAN paper](https://arxiv.org/abs/1703.10593) by Zhu et al. (2017).

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-optional-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

---

![horse_results](https://github.com/user-attachments/assets/5c197731-84f2-4863-b548-efdb54a30601)

*Pre-trained official weights were loaded successfully — confirming this is a **1-to-1 architecture match** with the original paper.*

</div>

---

## 📑 Table of Contents

- [✨ Highlights](#-highlights)
- [🏗️ Architecture](#️-architecture)
- [📁 Project Structure](#-project-structure)
- [⚙️ Configuration](#️-configuration)
- [🚀 Quick Start](#-quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Dataset Setup](#dataset-setup)
  - [Inference with Pre-trained Weights](#inference-with-pre-trained-weights)
- [🛠️ Training](#️-training)
- [📊 Results](#-results)
- [📚 References](#-references)

---

## ✨ Highlights

| Feature | Detail |
|---|---|
| 🔁 **Cycle-Consistent Loss** | Enforces `F(G(x)) ≈ x` without paired training data |
| 🏛️ **Faithful Architecture** | 1-to-1 match verified against official pre-trained weights |
| ⚡ **GPU / CPU Friendly** | Auto-detects CUDA; falls back to CPU transparently |
| 🧩 **Modular Code** | Generator & Discriminator decoupled in `Architecture/` |
| 🖼️ **Horse ↔ Zebra Demo** | Ready-to-run inference script with pre-trained models |

---

## 🏗️ Architecture

CycleGAN trains **two generators** and **two discriminators** simultaneously:

```
Domain X (Horse)          Domain Y (Zebra)
─────────────────         ─────────────────
    x ──► G_XY ──► ŷ ──► D_Y   (real vs fake zebra)
          │
          └──► F_YX(ŷ) ──► cycle loss with x

    y ──► G_YX ──► x̂ ──► D_X   (real vs fake horse)
          │
          └──► G_XY(x̂) ──► cycle loss with y
```

### Generator (`Architecture/Generator.py`)

```
Input (3×256×256)
    │
    ▼
7×7 Conv → InstanceNorm → ReLU          (initial block)
    │
    ▼  ×2
3×3 Conv ↓2 (Down-sampling blocks)
    │
    ▼  ×9
Residual Blocks (256 channels)
    │
    ▼  ×2
3×3 ConvTranspose ↑2 (Up-sampling blocks)
    │
    ▼
7×7 Conv → Tanh                          (output block)
    │
Output (3×256×256)
```

### Discriminator (`Architecture/Discriminator.py`)

A **PatchGAN** discriminator that classifies overlapping 70×70 image patches as real or fake — capturing local texture information far more effectively than a global decision.

```
Input (3×256×256)
    │
    ▼
4×4 Conv ↓2 → LeakyReLU(0.2)
    │
    ▼  ×3
4×4 Conv → InstanceNorm → LeakyReLU(0.2)
    │
    ▼
4×4 Conv → Sigmoid
    │
Patch output (1×30×30)
```

---

## 📁 Project Structure

```
CyclicGAN/
│
├── Architecture/
│   ├── Generator.py       # Generator with residual blocks
│   └── Discriminator.py   # PatchGAN Discriminator
│
├── pretrained_models/
│   ├── genh.pth.tar       # Pre-trained Horse→Zebra generator weights
│   └── 1.jpg              # Sample horse image for quick inference
│
├── Data.py                # HorseZebraDataset — PyTorch Dataset class
├── config.py              # All hyper-parameters & transforms
├── utils.py               # Checkpoint save/load, seeding helpers
├── load_&_infer.py        # Quick inference script
└── README.md
```

---

## ⚙️ Configuration

All knobs live in **`config.py`**:

| Variable | Default | Description |
|---|---|---|
| `DEVICE` | auto | `"cuda"` if GPU available, else `"cpu"` |
| `TRAIN_DIR` | `data/train` | Root directory for training data |
| `VAL_DIR` | `data/val` | Root directory for validation data |
| `BATCH_SIZE` | `1` | Training batch size |
| `LEARNING_RATE` | `1e-5` | Adam optimizer LR |
| `LAMBDA_CYCLE` | `10` | Weight for cycle-consistency loss |
| `LAMBDA_IDENTITY` | `0.0` | Weight for identity loss (0 = disabled) |
| `NUM_EPOCHS` | `10` | Number of training epochs |
| `NUM_WORKERS` | `4` | DataLoader worker threads |
| `LOAD_MODEL` | `False` | Resume from checkpoint? |
| `SAVE_MODEL` | `True` | Save checkpoints after each epoch? |

---

## 🚀 Quick Start

### Prerequisites

- Python ≥ 3.8
- pip

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/Wolfy024/CyclicGAN.git
cd CyclicGAN

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install torch torchvision matplotlib Pillow
```

### Dataset Setup

The model is trained on the **Horse2Zebra** dataset from the official CycleGAN project.

```bash
# Download & extract
wget https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip
unzip horse2zebra.zip -d data/
```

Expected layout after extraction:

```
data/
├── train/
│   ├── horses/   # trainA
│   └── zebras/   # trainB
└── val/
    ├── horses/   # testA
    └── zebras/   # testB
```

### Inference with Pre-trained Weights

```bash
python "load_&_infer.py"
```

This loads the official **Horse → Zebra** generator weights from `pretrained_models/genh.pth.tar` and displays the translated result for the sample image.

---

## 🛠️ Training

> ⚠️ A training script (`train.py`) is the natural next step. The modular design of this repo makes it straightforward to wire up — all building blocks are already in place.

**Skeleton to get you started:**

```python
from Architecture.Generator import Generator
from Architecture.Discriminator import Discriminator
from Data import HorseZebraDataset
import config

gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
disc_H = Discriminator(in_channel=3).to(config.DEVICE)
disc_Z = Discriminator(in_channel=3).to(config.DEVICE)

# ... define optimizers, losses (MSE + L1), and training loop
```

Key losses to implement:

```
L_total = L_adv(G, D) + λ_cycle * L_cycle + λ_identity * L_identity
```

---

## 📊 Results

The image below was generated by loading the **official pre-trained weights** into this implementation, confirming architectural fidelity:

![horse_results](https://github.com/user-attachments/assets/5c197731-84f2-4863-b548-efdb54a30601)

---

## 📚 References

- **CycleGAN Paper** — Zhu et al., 2017  
  [*Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*](https://arxiv.org/abs/1703.10593)
- **Official Implementation** — [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- **Horse2Zebra Dataset** — [Berkeley EECS](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/)

---

<div align="center">

Made with ❤️ by [Wolfy024](https://github.com/Wolfy024)

⭐ Star this repo if you found it helpful!

</div>
