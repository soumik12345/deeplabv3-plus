import os
from glob import glob
from typing import List, Optional

import tensorflow as tf
import wandb
from keras_cv.models import DeepLabV3Plus
from tensorflow import keras

from cityscapes_train.data import CityScapesSegmentationDataLoader
from cityscapes_train.utils import (
    fetch_artifact_creation_configs,
    parse_evaluation_args,
)


def main(model_artifact_address: str):
    wandb.init(project="deeplabv3-keras-cv", entity="geekyrakshit", job_type="evaluate")
    wandb.config = fetch_artifact_creation_configs(model_artifact_address)

    model_artifact = wandb.use_artifact(model_artifact_address, type="model")
    model_artifact_dir = model_artifact.download()
    with keras.utils.custom_object_scope({"DeepLabV3Plus": DeepLabV3Plus}):
        model = keras.saving.load_model(os.path.join(model_artifact_dir, "model.keras"))

    data_loader = CityScapesSegmentationDataLoader(
        artifact_address="geekyrakshit/semantic-segmentation-pipeline/cityscapes:latest",
        image_size=args.image_size,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
    )
    datasets = data_loader.build_datasets()

    for key, dataset in datasets.item():
        print(f"Evaluating split {key}...")
        loss, accuracy, iou = model.evaluate(dataset)
        wandb.log(
            {f"{key}/loss": loss, f"{key}/accuracy": accuracy, f"{key}/mean_io_u": iou},
            commit=False,
        )

    wandb.log({})
