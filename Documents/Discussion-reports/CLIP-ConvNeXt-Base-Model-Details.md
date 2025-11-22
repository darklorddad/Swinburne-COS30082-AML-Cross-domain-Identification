### 1. Image Classification (ImageNet-1k)
These metrics are taken directly from the "Model Comparison" tables found in the `timm` model cards.

| Hugging Face Repo Name | Top-1 Acc | Top-5 Acc | Resolution | Param Count |
| :--- | :--- | :--- | :--- | :--- |
| **`timm/convnextv2_base.fcmae_ft_in22k_in1k`** | **86.740%** | 98.022% | 224 x 224 | 88.72 M |
| **`timm/convnext_base.clip_laion2b_augreg_ft_in12k_in1k`** | **86.344%** | 97.970% | 256 x 256 | 88.59 M |
| **`timm/convnext_base.fb_in22k_ft_in1k`** | **85.822%** | 97.866% | 224 x 224 | 88.59 M |
| **`facebook/convnextv2-base-22k-224`** | *N/A* | *N/A* | 224 x 224 | 88.7 M |

**Note:** The repository `facebook/convnextv2-base-22k-224` text provided did not contain a performance metrics table, though it is the official checkpoint for the `timm/convnextv2...` model listed first.

### 2. Zero-Shot Classification
This metric is taken from the "Model Details" section of the `laion` model card.

| Hugging Face Repo Name | Zero-Shot Top-1 | Dataset | Augmentation |
| :--- | :--- | :--- | :--- |
| **`laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg`** | **71.5%** | LAION-2B | RRC, RE, SD* |

*\*Augmentation Key: RRC (Random Resize Crop), RE (Random Erasing), SD (Stochastic Depth)*