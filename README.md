# Cross-domain Plant Species Identification

## Project Overview
This project aims to solve the challenge of cross-domain plant species identification, specifically transferring knowledge from herbarium sheet images to field images. This is critical for identifying data-deficient plant species where field images are scarce but herbarium collections are available.

## Dataset
The dataset is derived from the PlantCLEF 2020 Challenge and consists of 100 species.
- **Training Data**: A mix of herbarium sheets and field images. Some classes lack field images entirely in the training set.
- **Test Data**: Exclusively field images.

### Dataset Balance
Below is the distribution of the dataset, highlighting the class imbalance and the proportion of herbarium vs. field images.

![Dataset Balance](path/to/your/dataset_balance_image.png)

## Methodology
We employ a deep learning approach using **ArcFace (Additive Angular Margin Loss)**. ArcFace is chosen for its ability to learn highly discriminative features, which is beneficial for handling:
1.  **Class Imbalance**: By enforcing a margin, it prevents the model from ignoring minority classes.
2.  **Domain Shift**: It encourages the model to learn class-specific features that are robust across the herbarium and field domains.

### Model Architecture
The architecture utilises a CNN backbone (e.g., ResNet or ViT) with an ArcFace head.

![ArcFace Model Architecture](path/to/your/arcface_architecture_image.png)

## Results and Visualisation
To evaluate the effectiveness of the embeddings learned by the model, we use t-SNE (t-Distributed Stochastic Neighbor Embedding) to visualise the high-dimensional features in 2D space.

### t-SNE Visualisation
The plot below shows the clustering of different plant species.

![t-SNE Visualisation](path/to/your/tsne_visualisation.png)

## Usage

### Training
To train the model using AutoTrain:
```bash
python -m autotrain.trainers.image_classification_custom --training_config config.json
```

### Inference
To run the inference application:
```bash
python inference_app/app.py
```
