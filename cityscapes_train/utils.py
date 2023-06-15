import argparse

import wandb

from .mappings import BACKBONE_DICT


def parse_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone_preset",
        type=str,
        required=True,
        choices=list(BACKBONE_DICT.keys()),
        help="KerasCV Backbone Preset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for TensorFlow",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Image size",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=34,
        help="Number of classes",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--initial_learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--end_learning_rate",
        type=float,
        default=1e-7,
        help="End learning rate",
    )
    parser.add_argument(
        "--lr_decay_power",
        type=float,
        default=0.99,
        help="Learning rate decay power",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight Decay",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs",
    )
    args = parser.parse_args()
    return args


def parse_evaluation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_artifact_address",
        type=str,
        required=True,
        help="Address of the Weights & Biases artifact corresponding to the model checkpoint.",
    )
    args = parser.parse_args()
    return args


def fetch_artifact_creation_configs(artifact_address):
    api = wandb.Api()
    artifact = api.artifact(artifact_address)
    run = artifact.logged_by()
    return run.config
