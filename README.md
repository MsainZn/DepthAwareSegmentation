# Towards Robust Breast Segmentation: Leveraging Depth Awareness and Convexity Optimization

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://github.com/TO_BE_PUBLISHED_AFTER_REVIEW/segmentation-framework)
[![GitHub](https://img.shields.io/badge/Code-GitHub-blue)](https://github.com/TO_BE_PUBLISHED_AFTER_REVIEW/segmentation-framework)

This repository contains the official implementation of the paper:  
**Towards Robust Breast Segmentation: Leveraging Depth Awareness and Convexity Optimization For Tackling Data Scarcity**

## 📄 Abstract
> Breast segmentation plays a critical role for objective pre‑ and post‑operative aesthetic evaluation but is challenged by limited data, class imbalance, and anatomical variability. We introduce an encoder–decoder framework with a Segment Anything Model (SAM) backbone, enhanced with synthetic depth maps and a multi‑term loss combining weighted cross‑entropy, convexity, and depth alignment constraints. Evaluated on a 120-patient dataset, our approach achieves a balanced test dice score of 98.75%—a 4.5% improvement over prior methods—with dice of 95.5% (breast) and 89.2% (nipple). Depth injection reduces noise and focuses on anatomical regions (+0.47% body, +1.04% breast dice), while convexity optimization improves nipple mask plausibility to 99.86%.

<p align="center">
  <img src="figs/arch.png" width="800" alt="Model Architecture">
</p>

## ✨ Key Features
- **SAM-Driven Encoding**: Leverages Segment Anything Model (SAM) backbone for robust feature extraction
- **Depth-Aware Segmentation**: Integrates synthetic depth maps from [Depth-Anything-v2](https://github.com/LiheYoung/Depth-Anything) to enhance geometric understanding
- **Convexity Optimization**: Novel loss formulation enforcing anatomical plausibility for nipple regions
- **Multi-Term Loss**: Combines weighted cross-entropy, depth alignment, and convexity constraints
- **Cross-Dataset Robustness**: Validated on CINDERELLA dataset with strong generalization

## 📊 Results
| Region  | DSC Score (%) | Improvement |
|---------|---------------|-------------|
| **Body**    | 98.7          | +0.47%      |
| **Breast**  | 95.5          | +1.04%      |
| **Nipple**  | 89.2          | +3% Conv    |
| **Mean**    | 98.75         | +4.5%       |

*Comparison with SoTA models:*
| Model       | Breast DSC | Nipple DSC |
|-------------|------------|------------|
| MobileNet   | 89.8%      | 78.1%      |
| Segformer   | 94.3%      | 88.8%      |
| **Ours**    | **95.5%**  | **89.2%**  |

## 🚀 Usage
*Code will be released upon paper acceptance. Planned functionality:*
```bash
# Installation
git clone https://github.com/TO_BE_PUBLISHED_AFTER_REVIEW/segmentation-framework
pip install -r requirements.txt

# Inference
python predict.py --input path/to/image.jpg --output segmentation.png

# Training
python train.py --config configs/base.yml --data-dir path/to/dataset

📂 Dataset
The Breast-Aesthetics (BA) dataset consists of:

120 RGB images with segmentation masks

Classes: Background (44.5%), Body (42.2%), Breast (12.4%), Nipple (1.0%)

Split: 70% train, 10% validation, 20% test

Access: Private dataset available upon request. Cross-validation performed on CINDERELLA dataset.

📍 Citation
bibtex
@article{zolfagharnasab2025towards,
  title={Towards Robust Breast Segmentation: Leveraging Depth Awareness and Convexity Optimization},
  author={Zolfagharnasab, Mohammad H. and Gonçalves, Tiago and Ferreira, Pedro and Cardoso, Maria J. and Cardoso, Jaime S.},
  journal={To appear},
  year={2025}
}

For questions or dataset access, contact: h.z.nasab@gmail.com
