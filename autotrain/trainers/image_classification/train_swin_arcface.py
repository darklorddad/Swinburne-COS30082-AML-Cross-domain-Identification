import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.state import PartialState
from huggingface_hub import HfApi
from datasets import load_from_disk, load_dataset
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from autotrain import logger
from autotrain.trainers.common import (
    ALLOW_REMOTE_CODE,
    LossLoggingCallback,
    TrainStartCallback,
    UploadLogs,
    pause_space,
    remove_autotrain_data,
    save_training_params,
)
from autotrain.trainers.image_classification import utils


# --- Custom ArcFace Module ---
class ArcFaceHead(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, sub_centers=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.sub_centers = sub_centers

        # Weight shape: (out_features, in_features) or (out_features*k, in_features)
        self.weight = nn.Parameter(torch.FloatTensor(out_features * sub_centers, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, label=None):
        # Normalize features and weights
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))

        if self.sub_centers > 1:
            # Sub-Center logic: Reshape to (Batch, Class, SubCenters) and take max
            cosine = cosine.view(-1, self.out_features, self.sub_centers)
            cosine, _ = torch.max(cosine, dim=2)

        if label is None:
            return cosine

        # Add margin
        # cos(theta + m)
        # Note: This is a simplified implementation for brevity.
        # Full implementation usually handles float precision and numerical stability guards.
        index = torch.where(label != -100)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine[index] -= m_hot

        return cosine * self.s


class ArcFaceModel(nn.Module):
    def __init__(self, config, num_labels, arcface_args):
        super().__init__()
        self.config = config
        # Load backbone without the classification head
        self.backbone = AutoModel.from_pretrained(
            config._name_or_path,
            config=config,
            trust_remote_code=ALLOW_REMOTE_CODE,
        )

        # Determine embedding dimension
        if hasattr(config, "hidden_size"):
            self.embedding_dim = config.hidden_size
        elif hasattr(config, "d_model"):
            self.embedding_dim = config.d_model
        elif hasattr(config, "hidden_sizes"):
            self.embedding_dim = config.hidden_sizes[-1]
        else:
            raise ValueError(
                "Could not determine embedding dimension from config. "
                "Please ensure the model config has 'hidden_size', 'd_model', or 'hidden_sizes'."
            )

        self.head = ArcFaceHead(
            in_features=self.embedding_dim,
            out_features=num_labels,
            s=arcface_args['s'],
            m=arcface_args['m'],
            sub_centers=arcface_args['sub_centers'],
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, pixel_values, labels=None):
        outputs = self.backbone(pixel_values=pixel_values)
        # Use pooled output (often pooler_output or last_hidden_state mean)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embeds = outputs.pooler_output
        else:
            # Fallback for models without pooler: mean of last hidden state
            hidden_state = outputs.last_hidden_state
            if len(hidden_state.shape) == 4:
                # CNN: (B, C, H, W) -> Global Average Pooling -> (B, C)
                embeds = hidden_state.mean(dim=[2, 3])
            else:
                # Transformer: (B, S, D) -> Mean Pooling -> (B, D)
                embeds = hidden_state.mean(dim=1)

        logits = self.head(embeds, labels)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits, "embeddings": embeds}

    def save_pretrained(self, save_directory):
        self.backbone.save_pretrained(save_directory)
        # Save head weights separately or relying on state_dict,
        # but for AutoTrain standard flow, we usually just save the backbone
        # unless we want to resume exact training.
        # For inference (retrieval), only backbone is needed.
        torch.save(self.head.state_dict(), f"{save_directory}/arcface_head.bin")


# --- Custom Trainer to handle the specific forward pass signature ---
class ArcFaceTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(pixel_values=inputs["pixel_values"], labels=labels)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss


def train(config):
    if config.data_path == f"{config.project_name}/autotrain-data":
        train_data = load_from_disk(config.data_path)[config.train_split]
    else:
        if ":" in config.train_split:
            dataset_config_name, split = config.train_split.split(":")
            train_data = load_dataset(
                config.data_path,
                name=dataset_config_name,
                split=split,
                token=config.token,
                trust_remote_code=ALLOW_REMOTE_CODE,
            )
        else:
            train_data = load_dataset(
                config.data_path,
                split=config.train_split,
                token=config.token,
                trust_remote_code=ALLOW_REMOTE_CODE,
            )

    valid_data = None
    if config.valid_split is not None:
        if config.data_path == f"{config.project_name}/autotrain-data":
            valid_data = load_from_disk(config.data_path)[config.valid_split]
        else:
            if ":" in config.valid_split:
                dataset_config_name, split = config.valid_split.split(":")
                valid_data = load_dataset(
                    config.data_path,
                    name=dataset_config_name,
                    split=split,
                    token=config.token,
                    trust_remote_code=ALLOW_REMOTE_CODE,
                )
            else:
                valid_data = load_dataset(
                    config.data_path,
                    split=config.valid_split,
                    token=config.token,
                    trust_remote_code=ALLOW_REMOTE_CODE,
                )

    # Determine Classes
    classes = train_data.features[config.target_column].names
    num_classes = len(classes)

    # Config
    model_config = AutoConfig.from_pretrained(config.model, trust_remote_code=ALLOW_REMOTE_CODE, token=config.token)

    # Initialize Custom Model
    model = ArcFaceModel(
        config=model_config,
        num_labels=num_classes,
        arcface_args={
            "s": config.arcface_s,
            "m": config.arcface_m,
            "sub_centers": config.sub_centers,
        },
    )

    # Image Processor
    image_processor = AutoImageProcessor.from_pretrained(
        config.model, token=config.token, trust_remote_code=ALLOW_REMOTE_CODE
    )

    # Process Data (Reuse existing utils)
    train_data, valid_data = utils.process_data(train_data, valid_data, image_processor, config)

    # Setup Arguments
    training_args = dict(
        output_dir=config.project_name,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.lr,
        num_train_epochs=config.epochs,
        save_strategy="epoch",
        eval_strategy="epoch" if valid_data else "no",
        logging_steps=config.logging_steps,
        save_total_limit=config.save_total_limit,
        remove_unused_columns=False,  # Important for custom forward signature
        label_names=["labels"],
        # Enable built-in CutMix/MixUp if supported by transformers version (optional but recommended)
        # torch_compile=True (Optimization)
    )

    args = TrainingArguments(**training_args)

    callbacks = [UploadLogs(config=config), LossLoggingCallback(), TrainStartCallback()]
    if config.valid_split:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience,
                early_stopping_threshold=config.early_stopping_threshold,
            )
        )

    trainer = ArcFaceTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        callbacks=callbacks,
    )

    trainer.train()

    # Save
    trainer.save_model(config.project_name)
    image_processor.save_pretrained(config.project_name)

    model_card = utils.create_model_card(config, trainer, num_classes)

    # save model card to output directory as README.md
    with open(f"{config.project_name}/README.md", "w") as f:
        f.write(model_card)

    if config.push_to_hub and config.username != "local":
        if PartialState().process_index == 0:
            remove_autotrain_data(config)
            save_training_params(config)
            logger.info("Pushing model to hub...")
            api = HfApi(token=config.token)
            api.create_repo(
                repo_id=f"{config.username}/{config.project_name}", repo_type="model", private=True, exist_ok=True
            )
            api.upload_folder(
                folder_path=config.project_name, repo_id=f"{config.username}/{config.project_name}", repo_type="model"
            )

    if PartialState().process_index == 0:
        pause_space(config)
