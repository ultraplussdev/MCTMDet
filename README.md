# MCTMDet

**MCTMDet** is a memory-enhanced one-stage video object detection framework built upon YOLOX.  
It introduces the following components:

- ✅ Multi-scale spatial context encoding  
- ✅ Dual-branch temporal modeling using Mamba  
- ✅ Class-aware memory fusion for inter-frame enhancement

---

## 🛠 Installation

### 1. Install YOLOX (base framework)

Before using MCTMDet, please make sure the base YOLOX framework is properly installed.

You can refer to the original YOLOX installation instructions in our repository here:  
👉 [YOLOX_README.md](./YOLOX_README.md)

If you already have YOLOX installed and functional, you may skip this step.
---

### 2. Install additional dependencies for MCTMDet

After installing YOLOX, run the following command to install additional dependencies required for MCTMDet:

```bash
pip install einops>=0.6 timm>=0.6.13 mamba-ssm==1.2.0 transformers>=4.30
```
## 📦 Pretrained Models

We provide the following pretrained model for evaluation:

- [MCTMdet_memory.pth (VID)](https://github.com/ultraplussdev/MCTMDet/releases/download/v1.0.0/MCTMDet_memory.pth)  
  This model is trained on ImageNet VID and used in the demo script  
  `tools/vid_mamba_demo.py` with the configuration file  
  `exps/MCTMDet/mamba_memory_use.py`.

## 🎬 Run Demo
To run inference on a video sequence, use the following command:

```bash
python tools/vid_mamba_demo.py \
  -f exps/MCTMDet/mamba_memory_use.py \
  -c MCTMdet_memory.pth \
  --path root/autodl-tmp/datasets/datasets/ILSVRC2015/Data/VID/snippets/val/ILSVRC2015_val_00007001.mp4 \
  --conf 0.25 \
  --nms 0.5 \
  --tsize 640 \
  --output_dir root/output_dir
```
## 📁 Project Structure
- `tools/vid_mamba_demo.py` — Inference demo for video input  
- `exps/MCTMDet/` — Experiment config files  
- `yolox/models/` — Custom modules including EncoderMamba, MemoryManager, etc.  
- `YOLOX_README.md` — Original YOLOX documentation (retained for installation reference)

## 📄 Notes

- The recommended runtime environment is **Linux (Ubuntu 20.04)** with **CUDA 11.8**.
- The project has been tested under the following environment:

  - OS: Ubuntu 20.04 / CUDA 11.8  
  - Python: 3.10  
  - PyTorch: 2.5.1  
