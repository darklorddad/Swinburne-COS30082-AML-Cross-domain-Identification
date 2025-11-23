**ViT-Only** strategy (No DINOv2, No Triplet, Single Model)

This strategy leverages the **Swin Transformer** architecture combined with **Masked Image Modeling** and **Angular Margin Loss**.

### The Strategy: "SwinV2-MAE with ArcFace"

#### 1. The Backbone: Swin Transformer V2 (Base or Large)
*   **Why this ViT?** Standard ViTs (like ViT-Base) process images in fixed $16 \times 16$ patches and can lose fine-grained details. The **Swin Transformer** uses "Shifted Windows." It creates a hierarchy (similar to a CNN) but retains the attention mechanism of a Transformer.
*   **The Advantage:** Plant identification relies on micro-structures (vein patterns, leaf margins) and macro-structures (branching). Swin V2 excels at capturing both simultaneously at high resolution.

#### 2. The Weights: MAE (Masked Autoencoder) Pre-training
*   **The Constraint:** You cannot use DINOv2.
*   **The Solution:** Initialize your Swin Transformer with **MAE** weights (pre-trained on ImageNet-22k).
*   **Why it wins:** MAE trains by masking out 75% of an image and forcing the ViT to reconstruct the missing parts.
    *   This teaches the model **Geometry and Structure** (e.g., "If I see a stem here, a leaf *must* be there"), rather than just matching colors.
    *   Since Herbarium sheets are "structural maps" of plants (but with wrong colors), an MAE-initialized model is naturally better at bridging the Herbarium-Field gap than a standard supervised model.

#### 3. The Loss Function: Sub-Center ArcFace
*   **The Constraint:** No Triplet Loss.
*   **The Solution:** **Sub-Center ArcFace**.
*   **How it works:** Standard ArcFace forces all images of "Species A" to a single point on a hypersphere. "Sub-Center" allows the model to create *K* sub-centers for one class.
    *   The model will automatically learn two clusters for "Species A": one for its **Dried** look and one for its **Fresh** look, but mathematically tie them to the same Class ID.
    *   This handles the "Two-Stream" problem inside a **Single Model** without needing complex pairs.

#### 4. The Training Key: "CutMix" & Resolution
*   **Resolution:** $384 \times 384$ (Swin V2 scales very well to higher resolutions).
*   **Augmentation:** **CutMix**.
    *   ViTs love CutMix. You cut a patch from a *Field* image and paste it onto a *Herbarium* image.
    *   **Why:** It forces the Self-Attention layers to attend to "Fresh" textures and "Dried" textures *in the same image* to predict the label. This prevents the model from biasing heavily toward one domain.

### Summary of the Stack

| Component | The Paper (2022) | Your ViT Strategy (Winner) | Why Yours Wins |
| :--- | :--- | :--- | :--- |
| **Architecture** | Inception-v4 (CNN) | **Swin Transformer V2-Base** | Hierarchical Attention captures veins + global shape better. |
| **Initialization**| ImageNet-1k (Supervised)| **MAE (ImageNet-22k)** | Learns "Structure" over "Color"; better for domain gaps. |
| **Loss** | Triplet Loss | **Sub-Center ArcFace** | Handles multi-modal (Dried/Fresh) classes within one model. |
| **Data Aug** | Random Crop | **CutMix** | Forces the single model to learn domain-invariant features. |


