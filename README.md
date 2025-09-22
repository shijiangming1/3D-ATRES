<div align="center">
<h1>3D-ATRES: Ambiguity-Tolerant Learning for 3D Referring Expression Segmentation</h1>

<a href=""><img src="https://img.shields.io/badge/arXiv-25xx.xx001-b31b1b" alt="arXiv"></a>
<a href=''><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Processed_Data-blue'></a>
<a href="" target="_blank"><img src="https://img.shields.io/badge/Checkpoint-Orange" alt="checkpoint"></a>
<a href="http://521661fbe9494e65b3805ad5028c5cc1.cloud.lanyun.net:10000/" target="_blank"><img src="https://img.shields.io/badge/🌐-Live_Demo-green" alt="Live Demo"></a>

[Jiangming Shi](xxx), [Chaoyang Li](), [Luosong Guo]()
</div>

![demo show](./assets/demo.gif)

## 📖 Overview

<div style="text-align: center;">
    <img src="assets/introduction.png" alt="Method Overview" width="100%">
</div>

**3D-ATRES** (3D Ambiguity-Tolerant Referring Expression Segmentation) is a novel framework designed to address the challenging task of **3D Referring Expression Segmentation (3D-RES)**. It enables precise segmentation of a target object within a 3D point cloud from a natural language description, with a dedicated focus on identifying and resolving **multi-source ambiguities** commonly encountered in real-world scenarios.

### ✨ Key Features
- **🔄 TR² Semantic Structurizer**: Parses free-form language into structured **Target–Relation–Referent** triples to eliminate **prompt ambiguity**
- **🧭 Normal-Aware Spatial Alignment**: Uses surface normal cues for robust, **viewpoint-invariant** geometric reasoning, addressing **spatial ambiguity**
- **📊 Annotation Ambiguity Penalty**: Employs a probabilistic learning mechanism to handle **noisy or inconsistent annotations** during training
- **✅ State-of-the-Art Performance**: Achieves leading results on benchmark datasets **ScanRefer** and **Multi3DRefer**

## 🧠 Model Architecture

We will show the model architecture when the paper is released.

## 📊 Pretrained Models

We provide several pretrained models hosted on Hugging Face. Performance metrics are summarized below:

### 📈 ScanRefer Dataset

| **Pretrained Model** | **Precision@0.5** | **Precision@0.7** | **mIoU** |
|:--------------------:|:-----------------:|:-----------------:|:--------:|
| 3D-ATRES (Base)      |       xxx      |       xxx       |   xxx  |
| 3D-ATRES (Large)     |     **xxx**     |     **xxx**     | **xxx**|
| 3D-ATRES (Ensemble)  |       xxx       |      xxx       |   xxx  |

### 📉 Multi3DRefer Dataset

| **Pretrained Model** | **Overall Acc.** | **Mean Acc.** | **Frequency-weighted IoU** |
|:--------------------:|:----------------:|:-------------:|:--------------------------:|
| 3D-ATRES (Base)      |       xxx       |      xxx     |            xxx            |
| 3D-ATRES (Large)     |     **xxx**     |    **xxx**   |          **xxx**          |

> 🔍 *More detailed results and analyses can be found in our paper.*

## 🌐 Live Demo

We have built an **[interactive online demo](http://521661fbe9494e65b3805ad5028c5cc1.cloud.lanyun.net:10000/)** where you can upload point clouds and referring expressions to see 3D-ATRES in action!

#### Example1
![demo show](./assets/example1.gif)
#### Example2
![demo show](./assets/example2.gif)
#### Example3
![demo show](./assets/example3.gif)

## 🚀 Getting Started

We will show the started steps when the code is released.

## 🙏 Acknowledgements

We extend our sincere gratitude to the following open-source projects and their contributors, which have been instrumental in our research:

- [ReferIt3D](https://github.com/referit3d/referit3d) - For the foundational 3D referring expression benchmark
- [ScanRefer](https://github.com/daveredrum/ScanRefer) - For the ScanRefer dataset and baseline implementations  
- [Multi3DRefer](https://github.com/xxzhou/Multi3DRefer) - For the multi-view 3D referring expression dataset
- [Open3D](http://www.open3d.org/) - For the powerful 3D data processing tools
- [Hugging Face](https://huggingface.co/) - For providing the platform to share our models and datasets

We also thank the researchers and developers in the 3D vision and language community for their valuable insights and contributions.

## 📜 Citation

If you use 3D-ATRES in your research or find our work helpful, please cite our paper:

```bibtex
@inproceedings{shi20253datres,
  title = {3D-ATRES: Ambiguity-Tolerant Learning for 3D Referring Expression Segmentation},
  author = {Shi, Jiangming and Li, Chaoyang and Guo, Luosong},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year = {2025},
  volume = {39},
  number = {12},
  pages = {12345--12353},
  publisher = {xxxx}
}
