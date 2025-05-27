# LNR: Label Noise Rebalancing
**Learning Imbalanced Data with Beneficial Label Noise**

**Authors**: Guangzheng Hu, Feng Liu, Mingming Gong, Guanghui Wang, Liuhua Peng

**Introduction**: This repository provides an implementation for the ICML 2025 paper: "[Learning Imbalanced Data with Beneficial Label Noise](https://icml.cc/virtual/2025/poster/46163)" based on [MiSLAS](https://github.com/dvlab-research/MiSLAS). LNR is a model-agnostic, simple, and efficient data-level method for step-wised and long-tailed imbalanced learning, which greatly improves recognition accuracy and model fairness simultaneously.

## Installation

**Install LNR**
```
git clone https://github.com/guangzhengh/LNR.git
cd LNR
pip install -r requirements.txt
```

**Dataset Preparation**
* [CIFAR-10 & CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
* [ImageNet](http://image-net.org/index)
* [iNaturalist 2018](https://github.com/visipedia/inat_comp/tree/master/2018)
* [Places](http://places2.csail.mit.edu/download.html)

Change the `data_path` in `config/*/*.yaml` accordingly.

## Training

**Finetune with LNR**:

To train a model for MiSLAS Stage-2 with LNR enhanced, run:

```
python train_stage2_lnr.py --cfg config\cifar100\cifar100_imb001_stage2_mislas.yaml
```

The saved folder (including logs and checkpoints) is organized as follows.
```
MiSLAS
├── saved
│   ├── modelname_date
│   │   ├── ckps
│   │   │   ├── current.pth.tar
│   │   │   └── model_best.pth.tar
│   │   └── logs
│   │       └── modelname.txt
│   ...   
```