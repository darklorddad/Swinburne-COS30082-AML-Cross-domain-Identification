import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import load_from_disk, load_dataset
from autotrain.trainers.common import (
    ALLOW_REMOTE_CODE,
    LossLoggingCallback,
    TrainStartCallback,
    UploadLogs,
    monitor,
)
from autotrain.trainers.image_classification import utils
from autotrain.trainers.image_classification.params import ImageClassificationParams


class ArcFaceHead(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, sub_centers=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.sub_centers = sub_centers
        self.weight = nn.Parameter(torch.FloatTensor(out_features * sub_centers, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, label=None):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        if self.sub_centers > 1:
            cosine = cosine.view(-1, self.out_features, self.sub_centers)
            cosine, _ = torch.max(cosine, dim=2)

        if label is None:
            return cosine

        index = torch.where(label != -100)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine[index] -= m_hot
        return cosine * self.s


class ArcFaceModel(nn.Module):
    def __init__(self, config, num_labels, arcface_args):
        super().__init__()
        self.config = config
        self.backbone = AutoModel.from_pretrained(
            config._name_or_path,
            config=config,
            trust_remote_code=ALLOW_REMOTE_CODE,
        )
        if hasattr(config, "hidden_size"):
            self.embedding_dim = config.hidden_size
        elif hasattr(config, "hidden_sizes"):
            self.embedding_dim = config.hidden_sizes[-1]
        else:
            raise ValueError("Could not determine embedding dimension from config")

        self.head = ArcFaceHead(
            in_features=self.embedding_dim,
            out_features=num_labels,
            s=arcface_args["s"],
            m=arcface_args["m"],
            sub_centers=arcface_args["sub_centers"],
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, pixel_values, labels=None):
        outputs = self.backbone(pixel_values=pixel_values)
        # Generic pooling strategy
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embeds = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            if len(outputs.last_hidden_state.shape) == 4:
                embeds = outputs.last_hidden_state.mean(dim=[-2, -1])
            else:
                embeds = outputs.last_hidden_state.mean(dim=1)
        else:
            if len(outputs[0].shape) == 4:
                embeds = outputs[0].mean(dim=[-2, -1])
            else:
                embeds = outputs[0].mean(dim=1)

        logits = self.head(embeds, labels)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}

    def save_pretrained(self, save_directory):
        self.backbone.save_pretrained(save_directory)


class ArcFaceTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(pixel_values=inputs["pixel_values"], labels=labels)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss


def train(config):
    if isinstance(config, dict):
        config = ImageClassificationParams(**config)

    train_data = None
    valid_data = None

    if config.data_path == f"{config.project_name}/autotrain-data":
        train_data = load_from_disk(config.data_path)[config.train_split]
        if config.valid_split:
            valid_data = load_from_disk(config.data_path)[config.valid_split]
    else:
        train_data = load_dataset(
            config.data_path,
            split=config.train_split,
            token=config.token,
            trust_remote_code=ALLOW_REMOTE_CODE,
        )
        if config.valid_split:
            valid_data = load_dataset(
                config.data_path,
                split=config.valid_split,
                token=config.token,
                trust_remote_code=ALLOW_REMOTE_CODE,
            )

    classes = train_data.features[config.target_column].names
    num_classes = len(classes)

    model_config = AutoConfig.from_pretrained(config.model, trust_remote_code=ALLOW_REMOTE_CODE, token=config.token)

    model = ArcFaceModel(
        config=model_config,
        num_labels=num_classes,
        arcface_args={
            "s": config.arcface_s,
            "m": config.arcface_m,
            "sub_centers": config.sub_centers,
        },
    )

    image_processor = AutoImageProcessor.from_pretrained(
        config.model, token=config.token, trust_remote_code=ALLOW_REMOTE_CODE
    )
    train_data, valid_data = utils.process_data(train_data, valid_data, image_processor, config)

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
        remove_unused_columns=False,
        label_names=["labels"],
        optim=config.optimizer,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        gradient_accumulation_steps=config.gradient_accumulation,
        fp16=config.mixed_precision == "fp16",
        bf16=config.mixed_precision == "bf16",
    )

    args = TrainingArguments(**training_args)
    callbacks = [LossLoggingCallback(), TrainStartCallback()]
    if config.push_to_hub and config.username != "local":
        callbacks.insert(0, UploadLogs(config=config))

    if config.valid_split:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience))

    trainer = ArcFaceTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(config.project_name)
    image_processor.save_pretrained(config.project_name)
