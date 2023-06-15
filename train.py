import argparse
import os
from glob import glob
from typing import List, Optional

import keras_cv
import tensorflow as tf
import wandb
from keras_cv.models import DeepLabV3Plus
from tensorflow import keras
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

from cityscapes_train.data import CityScapesSegmentationDataLoader
from cityscapes_train.mappings import BACKBONE_DICT
from cityscapes_train.utils import parse_train_args


def main(args):
    wandb.init(
        project="deeplabv3-keras-cv",
        entity="geekyrakshit",
        job_type="train",
        config=args,
    )

    tf.keras.utils.set_random_seed(args.seed)

    data_loader = CityScapesSegmentationDataLoader(
        artifact_address="geekyrakshit/semantic-segmentation-pipeline/cityscapes:latest",
        image_size=args.image_size,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
    )
    datasets = data_loader.build_datasets()

    backbone = BACKBONE_DICT[args.backbone_preset].from_preset(
        args.backbone_preset,
        input_shape=(args.image_size, args.image_size, 3),
    )
    model = DeepLabV3Plus(backbone=backbone, num_classes=args.num_classes)

    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    lr_schedule = keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=args.initial_learning_rate,
        decay_steps=tf.data.experimental.cardinality(datasets["train"]).numpy().item()
        * args.epochs,
        end_learning_rate=args.end_learning_rate,
        power=args.lr_decay_power,
    )
    optimizer = keras.optimizers.Adam(
        learning_rate=lr_schedule, weight_decay=args.weight_decay
    )

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=[
            "accuracy",
            keras.metrics.MeanIoU(
                num_classes=args.num_classes,
                sparse_y_pred=False,
                sparse_y_true=True,
            ),
        ],
        jit_compile=True,
    )
    model.fit(
        datasets["train"],
        validation_data=datasets["test"],
        epochs=args.epochs,
        callbacks=[
            WandbMetricsLogger(),
            WandbModelCheckpoint(filepath="model.keras"),
        ],
    )

    wandb.alert(title="Training Completed!!!", text="DeepLabV3+ training completed")


if __name__ == "__main__":
    args = parse_train_args()
    main(args)
