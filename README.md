# Cross-domain Plant Species Identification

### Project Overview
This project aims to solve the challenge of cross-domain plant species identification, specifically transferring knowledge from herbarium sheet images to field images. This is critical for identifying data-deficient plant species where field images are scarce but herbarium collections are available.

---

### Dataset
The dataset is derived from the PlantCLEF 2020 Challenge and consists of 100 species.
- **Training Data**: A mix of herbarium sheets and field images. Some classes lack field images entirely in the training set.
- **Test Data**: Exclusively field images.

| Dataset | Herbarium | Field | Total |
| :--- | :--- | :--- | :--- |
| **Train** | 3,700 | 1,044 | 4,744 |
| **Test** | - | 207 | 207 |

### Dataset Balance
Below is the distribution of the dataset, highlighting the class imbalance and the proportion of herbarium vs. field images.

![Dataset Balance](Dataset-PlantCLEF-2020-Challenge\Mix-set\Dataset-Class-Distribution.png)

---

### Methodology
We employ a deep learning approach using **ArcFace (Additive Angular Margin Loss)** combined with a **Class Balanced Sampler**.

1.  **ArcFace**: ArcFace is chosen for its ability to learn highly discriminative features. It encourages the model to learn class-specific features that are robust across the herbarium and field domains (Domain Shift).
2.  **Class Balanced Sampler**: To address the severe class imbalance in the dataset, we utilise a Class Balanced Sampler during training. This ensures that minority classes are sampled more frequently, preventing the model from being biased towards majority classes.

### Model Architecture
The architecture utilises a **ConvNeXt V2 Nano** backbone with an ArcFace head.

![ArcFace Model Architecture](Images\ArcFace-approach-architecture.png)

---

### Results and Visualisation

#### Performance Comparison
We compared the performance of the Standard model (Cross-Entropy Loss) and the ArcFace model using two evaluation methods:
1.  **Standard Classification**: Using the classification head (linear layer).
2.  **Prototype Retrieval**: Using nearest neighbour search with class prototypes computed from training embeddings.

![Prototype Retrieval Flow](Images/Prototype-Retrieval-Flow.png)

| Model | Method | MRR | Top-1 Accuracy | Top-5 Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **Standard (Normal)** | Standard | 0.548 | 47.8% | 61.4% |
| | Prototype | 0.568 | 48.3% | 65.7% |
| **ArcFace** | Standard | 0.544 (-0.004) | 47.3% (-0.5%) | 62.3% (+0.9%) |
| | Prototype | **0.569** (+0.001) | **50.2%** (+1.9%) | 64.7% (-1.0%) |

#### t-SNE Visualisation
To evaluate the effectiveness of the embeddings learned by the model, we use t-SNE (t-Distributed Stochastic Neighbor Embedding) to visualise the high-dimensional features in 2D space.

![Softmax vs ArcFace](Images/Softmax-vs-ArcFace.png)

The plots below show the clustering of different plant species, comparing the standard approach (top) with ArcFace (bottom).

![Standard t-SNE](Model-ConvNeXt-V2-Nano-A1-MH-62/t-SNE/tsne_plot.png)
![ArcFace t-SNE](Model-ConvNeXt-V2-Nano-A3-MH-64\t-SNE\tsne_plot.png)
