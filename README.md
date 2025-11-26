# Restora-Flow: Mask-Guided Image Restoration with Flow Matching

[![WACV 2026](https://img.shields.io/badge/WACV-2026-blue.svg)](https://wacv.thecvf.com/)
[![Paper](https://img.shields.io/badge/Paper-Link-green.svg)](https://arxiv.org/abs/2511.20152)

This repository contains the official implementation of our paper:

**_Restora-Flow: Mask-Guided Image Restoration with Flow Matching_**  
Accepted at **WACV 2026**.

<p align="center">
  <img src="figures/teaser.png" alt="Teaser" width="90%">
</p>

---

## 1. Setup

```bash
# Clone the repository
git clone <repo-url>
cd Restora-Flow

# Create a new conda environment
conda create -n restora-flow python=3.10 -y

# Activate the conda environment
conda activate restora-flow

# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install required packages
pip install -r requirements.txt
```

---

## 2. Natural Images

Supported datasets: CelebA, AFHQ-Cat, COCO

---

### 2.1 Download Datasets

#### **CelebA**
- Download: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset  
- Extract all contents into:
  ```
  natural/data/celeba
  ```

#### **AFHQ-Cat**
- Download: https://www.kaggle.com/datasets/dimensi0n/afhq-512/data
- Place test images into:
  ```
  natural/data/afhq_cat/test/cat
  ```

#### **COCO**
- Download: http://images.cocodataset.org/zips/val2017.zip  
- Place images into:
  ```
  natural/data/coco/val
  ```

> `natural/data` contains `.txt` files listing the image IDs used for evaluation in the paper.

---

### 2.2 Download Pretrained Checkpoints

Download all models:

```bash
cd natural
chmod +x download_all.sh
./download_all.sh
```

Or download individually:

```bash
cd natural
bash download.sh celeba-ot
bash download.sh celeba-ddpm
bash download.sh afhq-cat-ot
bash download.sh afhq-cat-ddpm
bash download.sh coco-ot
bash download.sh coco-ddpm
```

Checkpoints are stored in: ```natural/model_checkpoints/{dataset}/gaussian/{model_type}```

---

### 2.3 Running Natural Image Experiments
Before running experiments, set the repository base path in ```natural/src/dataloaders.py```:
```python 
base_path = "<absolute-path-to-Restora-Flow>"
```

Supported:
- **Model types:** `ot`, `ddpm`
- **Problems:** `denoising`, `box_inpainting`, `superresolution`, `random_inpainting`
- **Methods:** `repaint`, `ddnm`, `ot_ode`, `flow_priors`, `d_flow`, `pnp_flow`, `restora_flow`

#### Example: Restora-Flow on CelebA (box inpainting)

```bash
cd natural
python main.py --opts dataset celeba eval_split test model ot problem box_inpainting method restora_flow ode_steps 64 correction_steps 1 max_batch 1 batch_size_ip 1
```

#### Predefined experiments can be run with:

```
natural/script_test.sh
```

Results are saved to:

```
natural/results/{dataset}/{model_type}/{problem}/
```

---

## 3. Medical Images 

Supported datasets: X-ray Hand

---

### 3.1 Dataset Setup

1. Download the X-ray hand dataset from the original source: \
https://www.ipilab.org/Research/BAA/BAAindex.html

2. Place images into:
```
medical/dataset/xray_hand/images
```

3. (Optional) Adjust the image extension in:
```
medical/dataset.py
```

---

### 3.2 Download Pretrained Checkpoints

```bash
cd medical
chmod +x download.sh

bash download.sh xray-hand-flow
bash download.sh xray-hand-ddpm
```

Checkpoints are saved to: ```medical/model_checkpoints/xray_hand/{model_type}/full```

---

### 3.3 Running Medical Experiments

Supported:
- **Model types:** `flow`, `ddpm`
- **Problems:** `denoising`, `box_inpainting`, `superresolution`, `occlusion_removal`
- **Methods:** `RePaint`, `DDNM`, `OT-ODE`, `Flow-Priors`, `D-Flow`, `PnP-Flow`, `Restora-Flow`


```bash
cd medical
python main.py --model_type {flow|ddpm} --method {method} --problem {problem}
```

**Example:**

```bash
python main.py --model_type flow --method Restora-Flow --problem box_inpainting
```

Configuration options: ```medical/configs/restoration_config.yaml```

Output is saved to: ```medical/exports/hand_samples/{model_type}/{problem}/{method}/{timestamp}/```

---

## 4. Citation

```
@article{hadzic2025restoraflow,
  title={Restora-Flow: Mask-Guided Image Restoration with Flow Matching},
  author={Hadzic, Arnela and Thaler, Franz and Bogensperger, Lea and Joham, Simon Johannes and Urschler, Martin},
  journal={arXiv preprint arXiv:2511.20152},
  year={2025}
}
```

---

## 5. Acknowledgements

MedicalDataAugmentationTool taken from:  
https://github.com/christianpayer/MedicalDataAugmentationTool

GaussianDiffusion implementation based on:  
https://github.com/mobaidoctor/med-ddpm

Natural image framework builds upon:  
https://github.com/annegnx/PnP-Flow

CelebA and AFHQ-Cat flow models are from PnP-Flow.  
Other pretrained models were trained as part of this project.
