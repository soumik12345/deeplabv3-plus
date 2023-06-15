import os
from abc import ABC
from glob import glob
from typing import List, Optional

import tensorflow as tf
import wandb
from wandb_addons.utils import fetch_wandb_artifact


class SegmentationDataLoader(ABC):
    def __init__(
        self,
        image_size: int = 512,
        num_classes: Optional[int] = None,
        batch_size: int = 16,
    ):
        self.image_size = image_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        super().__init__()

    def random_crop(self, image, label):
        image_shape = tf.shape(image)[:2]
        crop_width = tf.random.uniform(
            shape=(),
            maxval=image_shape[1] - self.image_size + 1,
            dtype=tf.int32,
        )
        crop_height = tf.random.uniform(
            shape=(),
            maxval=image_shape[0] - self.image_size + 1,
            dtype=tf.int32,
        )
        image_cropped = image[
            crop_height : crop_height + self.image_size,
            crop_width : crop_width + self.image_size,
        ]
        label_cropped = label[
            crop_height : crop_height + self.image_size,
            crop_width : crop_width + self.image_size,
        ]
        return image_cropped, label_cropped

    def read_image(self, image_path, is_label: bool):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=1 if is_label else 3)
        image.set_shape([None, None, 1 if is_label else 3])
        image = tf.cast(image, dtype=tf.float32)
        image = image / 127.5 - 1 if not is_label else image
        return image

    def load_data(self, image, label):
        image = self.read_image(image, is_label=False)
        label = self.read_image(label, is_label=True)
        image = tf.image.resize(images=image, size=[self.image_size, self.image_size])
        label = tf.image.resize(images=label, size=[self.image_size, self.image_size])
        return image, label

    def get_dataset(self, image_list: List[str], label_list: List[str]):
        dataset = tf.data.Dataset.from_tensor_slices((image_list, label_list))
        dataset = dataset.map(
            map_func=self.load_data, num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        return dataset


class CityScapesSegmentationDataLoader(SegmentationDataLoader):
    def __init__(
        self,
        artifact_address: str,
        image_size: int = 512,
        num_classes: int | None = None,
        batch_size: int = 16,
    ):
        super().__init__(image_size, num_classes, batch_size)
        self.artifact_dir = fetch_wandb_artifact(
            artifact_address=artifact_address, artifact_type="dataset"
        )
        self.train_images, self.train_labels = self._prepare_train_split()
        self.val_images, self.val_labels = self._prepare_val_split()
        self.test_images, self.test_labels = self._prepare_test_split()

    def _prepare_train_split(self):
        train_images = sorted(
            glob(
                os.path.join(
                    self.artifact_dir,
                    os.path.join("leftImg8bit", "train", "*", "*.png"),
                )
            )
        )
        train_labels = sorted(
            glob(
                os.path.join(
                    self.artifact_dir,
                    os.path.join(
                        "gtFine_trainvaltest",
                        "gtFine",
                        "train",
                        "*",
                        "*labelIds.png",
                    ),
                )
            )
        )
        return train_images, train_labels

    def _prepare_val_split(self):
        val_images = sorted(
            glob(
                os.path.join(
                    self.artifact_dir, os.path.join("leftImg8bit", "val", "*", "*.png")
                )
            )
        )
        val_labels = sorted(
            glob(
                os.path.join(
                    self.artifact_dir,
                    os.path.join(
                        "gtFine_trainvaltest", "gtFine", "val", "*", "*labelIds.png"
                    ),
                )
            )
        )
        return val_images, val_labels

    def _prepare_test_split(self):
        test_images = sorted(
            glob(
                os.path.join(
                    self.artifact_dir, os.path.join("leftImg8bit", "test", "*", "*.png")
                )
            )
        )
        test_labels = sorted(
            glob(
                os.path.join(
                    self.artifact_dir,
                    os.path.join(
                        "gtFine_trainvaltest", "gtFine", "test", "*", "*labelIds.png"
                    ),
                )
            )
        )
        return test_images, test_labels

    def build_datasets(self):
        return {
            "train": self.get_dataset(self.train_images, self.train_labels),
            "val": self.get_dataset(self.val_images, self.val_labels),
            "test": self.get_dataset(self.test_images, self.test_labels),
        }
