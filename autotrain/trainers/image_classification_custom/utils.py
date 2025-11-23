import math
import os

import albumentations as A
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageFile
from sklearn import metrics

from autotrain.trainers.image_classification_custom.dataset import ImageClassificationDataset


ImageFile.LOAD_TRUNCATED_IMAGES = True


BINARY_CLASSIFICATION_EVAL_METRICS = (
    "eval_loss",
    "eval_accuracy",
    "eval_f1",
    "eval_auc",
    "eval_precision",
    "eval_recall",
)

MULTI_CLASS_CLASSIFICATION_EVAL_METRICS = (
    "eval_loss",
    "eval_accuracy",
    "eval_f1_macro",
    "eval_f1_micro",
    "eval_f1_weighted",
    "eval_precision_macro",
    "eval_precision_micro",
    "eval_precision_weighted",
    "eval_recall_macro",
    "eval_recall_micro",
    "eval_recall_weighted",
)

MODEL_CARD = """
---
tags:
- autotrain
- transformers
- image-classification{base_model}
widget:
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg
  example_title: Tiger
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/teapot.jpg
  example_title: Teapot
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/palace.jpg
  example_title: Palace{dataset_tag}
---

# Model Trained Using AutoTrain

- Problem type: Image Classification (Custom ArcFace)

## Validation Metrics
{validation_metrics}
"""


def _binary_classification_metrics(pred):
    raw_predictions, labels = pred
    # ArcFace outputs logits that are already scaled; argmax works fine
    predictions = np.argmax(raw_predictions, axis=1)
    result = {
        "f1": metrics.f1_score(labels, predictions),
        "precision": metrics.precision_score(labels, predictions),
        "recall": metrics.recall_score(labels, predictions),
        # AUC might be tricky with ArcFace logits, sticking to accuracy for now or need softmax
        # "auc": metrics.roc_auc_score(labels, raw_predictions[:, 1]),
        "accuracy": metrics.accuracy_score(labels, predictions),
    }
    return result


def _multi_class_classification_metrics(pred):
    raw_predictions, labels = pred
    predictions = np.argmax(raw_predictions, axis=1)
    results = {
        "f1_macro": metrics.f1_score(labels, predictions, average="macro"),
        "f1_micro": metrics.f1_score(labels, predictions, average="micro"),
        "f1_weighted": metrics.f1_score(labels, predictions, average="weighted"),
        "precision_macro": metrics.precision_score(labels, predictions, average="macro"),
        "precision_micro": metrics.precision_score(labels, predictions, average="micro"),
        "precision_weighted": metrics.precision_score(labels, predictions, average="weighted"),
        "recall_macro": metrics.recall_score(labels, predictions, average="macro"),
        "recall_micro": metrics.recall_score(labels, predictions, average="micro"),
        "recall_weighted": metrics.recall_score(labels, predictions, average="weighted"),
        "accuracy": metrics.accuracy_score(labels, predictions),
    }
    return results


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class ArcFaceClassifier(nn.Module):
    def __init__(self, model_name, num_classes, s=30.0, m=0.50, pretrained=True):
        super().__init__()
        # Create backbone without the classification head (num_classes=0)
        # global_pool='' ensures we get spatial features for CNNs to apply GeM

        # Sanitize model name for timm
        if "/" in model_name:
            model_name = model_name.split("/")[-1]

        candidates = [
            model_name,
            model_name.replace("-", "_"),
            model_name.replace("-", ""),
        ]

        self.backbone = None
        for name in candidates:
            try:
                self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="")
                break
            except Exception:
                pass

        if self.backbone is None:
            raise ValueError(f"Could not find a compatible timm model for {model_name}")

        # Auto-detect feature dimension
        try:
            if hasattr(self.backbone, "num_features"):
                feat_dim = self.backbone.num_features
            else:
                # Dummy forward pass to check
                dummy = torch.randn(1, 3, 224, 224)
                out = self.backbone(dummy)
                if len(out.shape) == 4:
                    feat_dim = out.shape[1]
                else:
                    feat_dim = out.shape[-1]
        except Exception:
            feat_dim = 768  # Fallback

        self.pooling = GeM()
        self.bn = nn.BatchNorm1d(feat_dim)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)

        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, pixel_values, labels=None):
        features = self.backbone(pixel_values)

        # Handle different backbone output shapes
        if len(features.shape) == 4:  # CNNs: (B, C, H, W)
            features = self.pooling(features).flatten(1)
        elif len(features.shape) == 3:  # Transformers: (B, N, C)
            features = features.mean(dim=1)
        elif len(features.shape) == 2:  # Already pooled (B, C)
            pass

        features = self.bn(features)

        # Calculate logits (cosine similarity)
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))

        # If labels are provided (Training or Eval)
        if labels is not None:
            if self.training:
                # Training: ArcFace Logic with Margin
                sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
                phi = cosine * self.cos_m - sine * self.sin_m
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)

                one_hot = torch.zeros(cosine.size(), device=pixel_values.device)
                one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
                output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
                output *= self.s
            else:
                # Eval: Standard Logits (scaled)
                output = cosine * self.s

            loss = self.loss_fn(output, labels)
            return {"loss": loss, "logits": output}

        # Inference (No labels): Return scaled logits
        return cosine * self.s


def process_data(train_data, valid_data, image_processor, config, model=None):
    """
    Processes training and validation data for image classification.

    Args:
        train_data (Dataset): The training dataset.
        valid_data (Dataset or None): The validation dataset. Can be None if no validation data is provided.
        image_processor (ImageProcessor): An object containing image processing parameters such as size, mean, and std.
        config (dict): Configuration dictionary containing additional parameters for dataset processing.
        model (nn.Module, optional): The model instance to resolve data config from.

    Returns:
        tuple: A tuple containing the processed training dataset and the processed validation dataset (or None if no validation data is provided).
    """
    # Resolve optimal normalization for the specific backbone
    try:
        if model is not None and hasattr(model, "backbone"):
            data_config = timm.data.resolve_data_config({}, model=model.backbone)
        else:
            data_config = timm.data.resolve_data_config({}, model=config.model)
        mean = data_config["mean"]
        std = data_config["std"]
        # Use image_processor size if available, else config default from timm
        if "input_size" in data_config:
            height, width = data_config["input_size"][1], data_config["input_size"][2]
        else:
            height, width = 224, 224
    except Exception:
        # Fallback to standard ImageNet
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        height, width = 224, 224

    train_transforms = A.Compose(
        [
            A.RandomResizedCrop(height=height, width=width),
            A.RandomRotate90(),
            A.HorizontalFlip(p=0.5),
            # Domain Erasing Augmentations (Configurable)
            A.ColorJitter(
                brightness=config.color_jitter_strength,
                contrast=config.color_jitter_strength,
                saturation=config.color_jitter_strength,
                hue=0.1,
                p=config.augment_prob,
            ),
            A.ToGray(p=config.grayscale_prob),
            A.Normalize(mean=mean, std=std),
        ]
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=height, width=width),
            A.Normalize(mean=mean, std=std),
        ]
    )
    train_data = ImageClassificationDataset(train_data, train_transforms, config)
    if valid_data is not None:
        valid_data = ImageClassificationDataset(valid_data, val_transforms, config)
        return train_data, valid_data
    return train_data, None


def create_model_card(config, trainer, num_classes):
    """
    Generates a model card for the given configuration and trainer.

    Args:
        config (object): Configuration object containing various settings.
        trainer (object): Trainer object used for model training and evaluation.
        num_classes (int): Number of classes in the classification task.

    Returns:
        str: A formatted string representing the model card.

    The function evaluates the model if a validation split is provided in the config.
    It then formats the evaluation scores based on whether the task is binary or multi-class classification.
    If no validation split is provided, it notes that no validation metrics are available.

    The function also checks the data path and model path in the config to determine if they are directories.
    Based on these checks, it formats the dataset tag and base model information accordingly.

    Finally, it uses the formatted information to create and return the model card string.
    """
    if config.valid_split is not None:
        eval_scores = trainer.evaluate()
        valid_metrics = (
            BINARY_CLASSIFICATION_EVAL_METRICS if num_classes == 2 else MULTI_CLASS_CLASSIFICATION_EVAL_METRICS
        )
        eval_scores = [f"{k[len('eval_'):]}: {v}" for k, v in eval_scores.items() if k in valid_metrics]
        eval_scores = "\n\n".join(eval_scores)

    else:
        eval_scores = "No validation metrics available"

    if config.data_path == f"{config.project_name}/autotrain-data" or os.path.isdir(config.data_path):
        dataset_tag = ""
    else:
        dataset_tag = f"\ndatasets:\n- {config.data_path}"

    if os.path.isdir(config.model):
        base_model = ""
    else:
        base_model = f"\nbase_model: {config.model}"

    model_card = MODEL_CARD.format(
        dataset_tag=dataset_tag,
        validation_metrics=eval_scores,
        base_model=base_model,
    )
    return model_card
