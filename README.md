# Cross-domain Plant Species Identification

### Project Overview
This project aims to solve the challenge of cross-domain plant species identification, specifically transferring knowledge from herbarium sheet images to field images. This is critical for identifying data-deficient plant species where field images are scarce but herbarium collections are available.

---

### Dataset
The dataset is derived from the PlantCLEF 2020 Challenge and consists of 100 species.
- **Training Data**: A mix of herbarium sheets and field images. Some classes lack field images entirely in the training set.
- **Test Data**: Exclusively field images.

### Dataset Balance
Below is the distribution of the dataset, highlighting the class imbalance and the proportion of herbarium vs. field images.

![Dataset Balance](Dataset-PlantCLEF-2020-Challenge\Mix-set\Dataset-Class-Distribution.png)

---

### Methodology
We employ a deep learning approach using **ArcFace (Additive Angular Margin Loss)**. ArcFace is chosen for its ability to learn highly discriminative features, which is beneficial for handling:
1.  **Class Imbalance**: By enforcing a margin, it prevents the model from ignoring minority classes.
2.  **Domain Shift**: It encourages the model to learn class-specific features that are robust across the herbarium and field domains.

### Model Architecture
The architecture utilises a **ConvNeXt V2 Nano** backbone with an ArcFace head.

![ArcFace Model Architecture](Images\ArcFace-approach-architecture.png)

---

### Results and Visualisation
To evaluate the effectiveness of the embeddings learned by the model, we use t-SNE (t-Distributed Stochastic Neighbor Embedding) to visualise the high-dimensional features in 2D space.

![Softmax vs ArcFace](Images/Softmax-vs-ArcFace.png)

### t-SNE Visualisation
The plots below show the clustering of different plant species, comparing the standard approach (top) with ArcFace (bottom).

![Standard t-SNE](Model-ConvNeXt-V2-Nano-A1-MH-62/t-SNE/tsne_plot.png)
![ArcFace t-SNE](Model-ConvNeXt-V2-Nano-A3-MH-64\t-SNE\tsne_plot.png)
