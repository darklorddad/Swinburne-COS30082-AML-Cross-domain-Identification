import argparse
import json
import os
import shutil

import numpy as np
import torch
from accelerate.state import PartialState
from safetensors.torch import save_file as safe_save_file
from datasets import load_dataset, load_from_disk
from huggingface_hub import HfApi
from transformers import (
    AutoImageProcessor,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import PrinterCallback

from autotrain import logger
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
from autotrain.trainers.image_classification_custom import utils
from autotrain.trainers.image_classification_custom.utils import ArcFaceClassifier
from autotrain.trainers.image_classification_custom.params import ImageClassificationParams


class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, custom_config=None, image_processor=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.custom_config = custom_config
        self.image_processor = image_processor

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        if self.class_weights is None:
            return super().get_train_dataloader()

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        # Access the underlying HF dataset from ImageClassificationDataset wrapper
        labels = train_dataset.data.with_format("numpy")[train_dataset.config.target_column]
        sample_weights = self.class_weights[torch.from_numpy(labels)]

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )

        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir

        os.makedirs(output_dir, exist_ok=True)

        # Save weights in safetensors format
        safe_save_file(self.model.state_dict(), os.path.join(output_dir, "model.safetensors"))

        # Save config.json
        if self.custom_config:
            with open(os.path.join(output_dir, "config.json"), "w") as f:
                json.dump(self.custom_config, f, indent=4)

        # Save training args
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # Save image processor
        if self.image_processor:
            self.image_processor.save_pretrained(output_dir)

        # Copy modeling files to output directory (essential for checkpoints)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for filename in ["configuration_arcface.py", "modeling_arcface.py"]:
            src = os.path.join(script_dir, filename)
            dst = os.path.join(output_dir, filename)
            if os.path.exists(src):
                shutil.copy(src, dst)


def parse_args():
    # get training_config.json from the end user
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", type=str, required=True)
    return parser.parse_args()


@monitor
def train(config):
    if isinstance(config, dict):
        config = ImageClassificationParams(**config)

    logger.info(f"Training configuration: {config}")

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

    logger.info(f"Train data: {train_data}")
    logger.info(f"Valid data: {valid_data}")

    classes = train_data.features[config.target_column].names
    logger.info(f"Classes: {classes}")
    label2id = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)

    if num_classes < 2:
        raise ValueError("Invalid number of classes. Must be greater than 1.")

    if config.valid_split is not None:
        num_classes_valid = len(valid_data.unique(config.target_column))
        if num_classes_valid != num_classes:
            raise ValueError(
                f"Number of classes in train and valid are not the same. Training has {num_classes} and valid has {num_classes_valid}"
            )

    # --- Initialize Custom ArcFace Model ---
    logger.info(f"Initializing custom ArcFace model with backbone: {config.model}")
    model = ArcFaceClassifier(
        model_name=config.model,
        num_classes=num_classes,
        s=config.arcface_s,
        m=config.arcface_m,
    )

    # Image Processor (mostly for consistency/saving, logic is in utils)
    try:
        image_processor = AutoImageProcessor.from_pretrained(config.model, token=config.token)
    except Exception:
        image_processor = None

    train_data, valid_data = utils.process_data(train_data, valid_data, image_processor, config, model=model)

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

    logger.info(f"Logging steps: {logging_steps}")

    scheduler = config.scheduler
    if scheduler == "cosine_warmup":
        scheduler = "cosine"

    training_args_dict = dict(
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
        remove_unused_columns=False,
    )

    if config.valid_split is not None:
        training_args_dict["eval_steps"] = logging_steps
        training_args_dict["save_steps"] = logging_steps

    if config.mixed_precision == "fp16":
        training_args_dict["fp16"] = True
    if config.mixed_precision == "bf16":
        training_args_dict["bf16"] = True

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

    # --- STEP 1: WARMUP (Linear Probe) ---
    logger.info("--- STEP A: WARMUP (Freezing Backbone) ---")
    for param in model.backbone.parameters():
        param.requires_grad = False

    warmup_args_dict = training_args_dict.copy()
    warmup_args_dict["num_train_epochs"] = config.warmup_epochs
    warmup_args_dict["learning_rate"] = config.head_lr
    warmup_args_dict["output_dir"] = os.path.join(config.project_name, "warmup")

    # Calculate class weights for sampler if enabled
    class_weights = None
    if config.use_class_balanced_sampler:
        import numpy as np

        labels = train_data.data[config.target_column]
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        logger.info("Using Class Balanced Sampler")

    # Prepare custom config for saving
    if image_processor is not None and hasattr(image_processor, "size"):
        image_size = image_processor.size
    elif image_processor is not None and hasattr(image_processor, "crop_size"):
        image_size = image_processor.crop_size
    else:
        image_size = {"height": 224, "width": 224}

    custom_config = {
        "architectures": ["ArcFaceClassifier"],
        "model_type": "custom_arcface",
        "auto_map": {
            "AutoConfig": "configuration_arcface.ArcFaceConfig",
            "AutoModelForImageClassification": "modeling_arcface.ArcFaceClassifier",
        },
        "backbone": config.model,
        "num_classes": num_classes,
        "arcface_s": config.arcface_s,
        "arcface_m": config.arcface_m,
        "image_size": image_size,
        "id2label": {i: c for i, c in enumerate(classes)},
        "label2id": label2id,
    }

    # Copy modeling files to output directory for AutoModel compatibility
    # We assume these files are located in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for filename in ["configuration_arcface.py", "modeling_arcface.py"]:
        src = os.path.join(script_dir, filename)
        dst = os.path.join(config.project_name, filename)
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            logger.warning(f"Could not find {filename} in {script_dir}. AutoModel loading might fail.")


    trainer_warmup = CustomTrainer(
        class_weights=class_weights,
        custom_config=custom_config,
        image_processor=image_processor,
        args=TrainingArguments(**warmup_args_dict),
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
    )
    trainer_warmup.train()

    # --- STEP 2: FINE-TUNING (Unfreeze) ---
    logger.info("--- STEP B: FINE-TUNING (Unfreezing Backbone) ---")
    for param in model.backbone.parameters():
        param.requires_grad = True

    logger.info(f"Using backbone_lr: {config.backbone_lr}")
    logger.info(f"Using head_lr: {config.head_lr}")
    logger.info(f"Using optimizer: {config.optimizer}")

    # Differential Learning Rates
    backbone_params = list(map(id, model.backbone.parameters()))
    head_params = filter(lambda p: id(p) not in backbone_params, model.parameters())

    optimizer_cls = torch.optim.AdamW
    if config.optimizer == "adam":
        optimizer_cls = torch.optim.Adam
    elif config.optimizer == "sgd":
        optimizer_cls = torch.optim.SGD

    optimizer = optimizer_cls(
        [
            {"params": model.backbone.parameters(), "lr": config.backbone_lr},
            {"params": head_params, "lr": config.head_lr},
        ],
        weight_decay=config.weight_decay,
    )

    args = TrainingArguments(**training_args_dict)

    trainer = CustomTrainer(
        class_weights=class_weights,
        custom_config=custom_config,
        image_processor=image_processor,
        args=args,
        model=model,
        callbacks=callbacks_to_use,
        compute_metrics=(
            utils._binary_classification_metrics if num_classes == 2 else utils._multi_class_classification_metrics
        ),
        train_dataset=train_data,
        eval_dataset=valid_data,
        optimizers=(optimizer, None),
    )
    trainer.remove_callback(PrinterCallback)
    trainer.train()

    logger.info("Finished training, saving model...")
    trainer.save_model(config.project_name)

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


if __name__ == "__main__":
    _args = parse_args()
    training_config = json.load(open(_args.training_config))
    _config = ImageClassificationParams(**training_config)
    train(_config)
