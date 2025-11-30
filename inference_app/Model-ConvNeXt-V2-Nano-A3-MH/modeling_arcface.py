import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutput
import math
import timm
from .configuration_arcface import ArcFaceConfig

class ArcFaceClassifier(PreTrainedModel):
    config_class = ArcFaceConfig

    def __init__(self, config):
        super().__init__(config)

        self.s = config.arcface_s
        self.m = config.arcface_m
        self.k = config.arcface_k
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        self.loss_fn = nn.CrossEntropyLoss()

        # Create backbone
        # We assume pretrained=False because we are loading weights from safetensors/bin immediately after
        self.backbone = timm.create_model(config.backbone, pretrained=False, num_classes=0, global_pool="avg")

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

        self.bn = nn.BatchNorm1d(feat_dim)
        self.weight = nn.Parameter(torch.FloatTensor(config.num_classes * self.k, feat_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, pixel_values, labels=None, output_hidden_states=None, return_dict=None):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        raw_features = self.backbone(pixel_values)
        features = raw_features

        # Handle different backbone output shapes
        if len(features.shape) == 4:  # CNNs: (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
            features = features.flatten(1)
        elif len(features.shape) == 3:  # Transformers: (B, N, C) -> (B, C)
            if features.shape[1] != 1 and features.shape[1] != features.shape[-1]:
                features = features.mean(dim=1)

        if len(features.shape) > 2:
            features = features.flatten(1)

        features = self.bn(features)

        # Calculate logits (cosine similarity)
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))

        if self.k > 1:
            cosine = torch.reshape(cosine, (-1, self.weight.shape[0] // self.k, self.k))
            cosine, _ = torch.max(cosine, axis=2)

        loss = None
        logits = cosine * self.s

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
                logits = output

            loss = self.loss_fn(logits, labels)

        if not return_dict:
            if labels is not None:
                return {"loss": loss, "logits": logits}
            return logits

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=(raw_features,) if output_hidden_states else None,
            pooler_output=features if output_hidden_states else None
        )
