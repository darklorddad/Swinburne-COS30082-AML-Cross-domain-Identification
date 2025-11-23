import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.state import PartialState
from huggingface_hub import HfApi
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from transformers.trainer_callback import PrinterCallback
from datasets import load_from_disk, load_dataset
from autotrain.trainers.common import (
    ALLOW_REMOTE_CODE,
    LossLoggingCallback,
    TrainStartCallback,
    UploadLogs,
    monitor,
    pause_space,
    remove_autotrain_data,
    save_training_params,
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

        if len(embeds.shape) > 2:
            embeds = embeds.flatten(start_dim=1)

        logits = self.head(embeds, labels)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}

    def save_pretrained(self, save_directory):
        self.backbone.save_pretrained(save_directory)


class ArcFaceTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(pixel_values=inputs["pixel_values"], labels=labels)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss


def train(config):
    if isinstance(config, dict):
        config = ImageClassificationParams(**config)

    valid_data = None
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

    classes = train_data.features[config.target_column].names
    num_classes = len(classes)
    label2id = {c: i for i, c in enumerate(classes)}

    if num_classes < 2:
        raise ValueError("Invalid number of classes. Must be greater than 1.")

    if config.valid_split is not None:
        num_classes_valid = len(valid_data.unique(config.target_column))
        if num_classes_valid != num_classes:
            raise ValueError(
                f"Number of classes in train and valid are not the same. Training has {num_classes} and valid has {num_classes_valid}"
            )

    model_config = AutoConfig.from_pretrained(
        config.model,
        num_labels=num_classes,
        trust_remote_code=ALLOW_REMOTE_CODE,
        token=config.token,
    )
    model_config._num_labels = len(label2id)
    model_config.label2id = label2id
    model_config.id2label = {v: k for k, v in label2id.items()}

    model = ArcFaceModel(
        config=model_config,
        num_labels=num_classes,
        arcface_args={
            "s": config.arcface_s,
            "m": config.arcface_m,
            "sub_centers": max(1, config.sub_centers),
        },
    )

    image_processor = AutoImageProcessor.from_pretrained(
        config.model, token=config.token, trust_remote_code=ALLOW_REMOTE_CODE
    )
    train_data, valid_data = utils.process_data(train_data, valid_data, image_processor, config)

    if config.logging_steps == -1:
        if config.valid_split is not None:
            logging_steps = int(0.2 * len(valid_data) / config.batch_size)
        else:
            logging_steps = int(0.2 * len(train_data) / config.batch_size)
        if logging_steps == 0:
            logging_steps = 1
        if logging_steps > 25:
            logging_steps = 25
        config.logging_steps = logging_steps
    else:
        logging_steps = config.logging_steps

    scheduler = config.scheduler
    if scheduler == "cosine_warmup":
        scheduler = "cosine"

    training_args = dict(
        output_dir=config.project_name,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=2 * config.batch_size,
        learning_rate=config.lr,
        num_train_epochs=config.epochs,
        eval_strategy=config.eval_strategy if config.valid_split is not None else "no",
        logging_steps=logging_steps,
        save_total_limit=config.save_total_limit,
        save_strategy=config.eval_strategy if config.valid_split is not None else "no",
        gradient_accumulation_steps=config.gradient_accumulation,
        report_to=config.log,
        auto_find_batch_size=config.auto_find_batch_size,
        lr_scheduler_type=scheduler,
        optim=config.optimizer,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        push_to_hub=False,
        load_best_model_at_end=True if config.valid_split is not None else False,
        ddp_find_unused_parameters=False,
    )

    if config.valid_split is not None:
        training_args["eval_steps"] = logging_steps
        training_args["save_steps"] = logging_steps

    if config.mixed_precision == "fp16":
        training_args["fp16"] = True
    if config.mixed_precision == "bf16":
        training_args["bf16"] = True

    if config.valid_split is not None:
        early_stop = EarlyStoppingCallback(
            early_stopping_patience=config.early_stopping_patience,
            early_stopping_threshold=config.early_stopping_threshold,
        )
        callbacks_to_use = [early_stop]
    else:
        callbacks_to_use = []

    callbacks_to_use.extend([LossLoggingCallback(), TrainStartCallback()])
    if config.push_to_hub and config.username != "local":
        callbacks_to_use.insert(0, UploadLogs(config=config))

    args = TrainingArguments(**training_args)
    trainer = ArcFaceTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        callbacks=callbacks_to_use,
        compute_metrics=(
            utils._binary_classification_metrics if num_classes == 2 else utils._multi_class_classification_metrics
        ),
    )
    trainer.remove_callback(PrinterCallback)
    trainer.train()

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
            api = HfApi(token=config.token)
            api.create_repo(
                repo_id=f"{config.username}/{config.project_name}", repo_type="model", private=True, exist_ok=True
            )
            api.upload_folder(
                folder_path=config.project_name, repo_id=f"{config.username}/{config.project_name}", repo_type="model"
            )

    if PartialState().process_index == 0:
        pause_space(config)
