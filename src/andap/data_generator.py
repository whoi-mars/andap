import tensorflow as tf
import numpy as np
from glob import glob
import random
import skimage
import os

from . import utils


class DataGenAugment(tf.keras.utils.Sequence):
    """
    Data generator with random cropping and flipping for data augmentation.
    """

    def __init__(self, path, batch_size=64, shuffle=True, rng_seed=0, **kwargs):
        """
        Initializes data generator used for training.

        Args:
            path (str): Path to training dataset.
            batch_size (int, optional): Number of samples per batch. Defaults to 64.
            shuffle (bool, optional): If True (default), samples are shuffled after each epoch.
            rng_seed (int, optional): Seed for random number generators. Defaults to 0.
        """
        super().__init__(**kwargs)

        self.rng = random.Random(rng_seed)
        self.np_rng = np.random.default_rng(rng_seed)

        self.path = path
        # Width and height of the models input image
        self.wh = 64
        self.downsamples = 4
        self.downsamplesN = 16
        self.batch_size = batch_size
        self.filenames = sorted(
            [
                os.path.split(f)[1].split(".")[-2]
                for f in glob(os.path.join(self.path, "*.png"))
            ]
        )
        if shuffle:
            self.rng.shuffle(self.filenames)
        self.shuffle = shuffle

        self.nChannels = 3
        self.img_format = 128

    def __len__(self):
        return int(np.ceil(len(self.filenames) * 4 / float(self.batch_size)))

    def __getitem__(self, idx):
        actual_bs = min(
            self.batch_size, len(self.filenames) * 4 - idx * self.batch_size
        )
        grid = np.zeros(
            (actual_bs, self.downsamplesN, self.downsamplesN, 8), dtype=np.float32
        )
        grid[..., 0] = -np.inf  # o
        grid[..., 1] = 0.5      # x
        grid[..., 2] = 0.5      # y

        X = np.zeros((actual_bs, self.wh, self.wh, self.nChannels))

        for b in range(actual_bs):
            file_idx = (idx * self.batch_size + b) % len(self.filenames)
            # Ellipse data: x,y,w,h,a,c
            ellipse_data = np.load(
                os.path.join(self.path, self.filenames[file_idx] + ".npy")
            )

            full_img = skimage.io.imread(
                os.path.join(self.path, self.filenames[file_idx] + ".png")
            )

            # Crop
            crop_x, crop_y = np.floor(
                self.np_rng.random((2)) * (self.img_format - self.wh)
            )
            crop_x = int(crop_x)
            crop_y = int(crop_y)
            X[b] = full_img[crop_y : crop_y + self.wh, crop_x : crop_x + self.wh]

            if len(ellipse_data) > 0:
                contained_mask = (
                    (ellipse_data[:, 0] > crop_x)
                    & (ellipse_data[:, 0] < crop_x + self.wh)
                    & (ellipse_data[:, 1] > crop_y)
                    & (ellipse_data[:, 1] < crop_y + self.wh)
                )
                ellipse_data[:, 0] -= crop_x
                ellipse_data[:, 1] -= crop_y

            # flip x-axis:
            if self.np_rng.random() > 0.5:
                X[b] = np.fliplr(X[b])
                if len(ellipse_data) > 0:
                    ellipse_data[:, 0] = self.wh - 1 - ellipse_data[:, 0]
                    ellipse_data[:, 4] *= -1

            # flip y-axis:
            if self.np_rng.random() > 0.5:
                X[b] = np.flipud(X[b])
                if len(ellipse_data) > 0:
                    ellipse_data[:, 1] = self.wh - 1 - ellipse_data[:, 1]
                    ellipse_data[:, 4] *= -1

            if len(ellipse_data) > 0:
                # Mask grid cells whose center is within an ellipse so they can
                # be ignored in objectness loss
                cell_x, cell_y = np.meshgrid(
                    np.arange(0, self.wh, self.downsamples) + self.downsamples / 2,
                    np.arange(0, self.wh, self.downsamples) + self.downsamples / 2,
                )
                for ed in ellipse_data:
                    grid[b, ..., 7] += utils.points_in_ellipse(
                        cell_x, cell_y, *ed[:4], np.deg2rad(ed[4])
                    )

                grid[b, ..., 7] = grid[b, ..., 7] > 0

                ellipse_data = ellipse_data[contained_mask]

                for i, ed in enumerate(ellipse_data):
                    if ed.shape[0] == 0:
                        continue

                    pIdX = int(ed[0] // self.downsamples)
                    pIdY = int(ed[1] // self.downsamples)

                    grid[b, pIdY, pIdX, 0] = np.inf
                    grid[b, pIdY, pIdX, 1] = (
                        ed[0] - pIdX * self.downsamples
                    ) / self.downsamples
                    grid[b, pIdY, pIdX, 2] = (
                        ed[1] - pIdY * self.downsamples
                    ) / self.downsamples
                    grid[b, pIdY, pIdX, 3] = (
                        min(ed[2], ed[3]) / self.downsamples
                    )  # width
                    grid[b, pIdY, pIdX, 4] = (
                        max(ed[2], ed[3]) / self.downsamples
                    )  # height
                    a = ((ed[4]) % 180) if ed[2] < ed[3] else ((ed[4] - 90) % 180)
                    a = a / 180
                    grid[b, pIdY, pIdX, 5] = a
                    if ed[5] == 1:
                        grid[b, pIdY, pIdX, 6] = np.inf
                    else:
                        grid[b, pIdY, pIdX, 6] = -np.inf

        return tf.convert_to_tensor(X, dtype=tf.float32), grid

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.filenames)

    def get_imbalance(self):
        """
        Get imbalance of samples with/without objects and with lying/standing penguins.

        Returns:
            float: Fraction of penguins in training data that are lying.
            float: Fraction of grid cells that contain a penguin.
        """
        n_obj = 0
        n_noobj = 0
        n_standing = 0
        n_lying = 0

        for filename in self.filenames:
            ellipse_data = np.load(os.path.join(self.path, filename + ".npy"))
            contained_mask = (
                (ellipse_data[:, 0] > 0)
                & (ellipse_data[:, 0] < self.img_format)
                & (ellipse_data[:, 1] > 0)
                & (ellipse_data[:, 1] < self.img_format)
            )
            ellipse_data = ellipse_data[contained_mask]
            if ellipse_data.shape[0] == 0:
                n_noobj += (self.img_format // self.downsamples) ** 2
                continue
            n_obj += len(ellipse_data)
            n_noobj += (self.img_format // self.downsamples) ** 2 - len(ellipse_data)
            n_standing += np.sum(ellipse_data[:, 5] == utils.posture_map["standing"])
            n_lying += np.sum(ellipse_data[:, 5] == utils.posture_map["lying"])

        return n_lying / n_obj, n_obj / (n_obj + n_noobj)


class DataGen(tf.keras.utils.Sequence):
    """
    Data Generator without augmentation for validation data.
    """

    def __init__(self, path, batch_size=64, rng_seed=0, **kwargs):
        """
        Initializes data generator used for validation.

        Args:
            path (str): Path to validation dataset.
            batch_size (int, optional): Number of samples per batch. Defaults to 64.
            rng_seed (int, optional): Seed for random number generators. Defaults to 0.
        """
        super().__init__(**kwargs)

        self.rng = random.Random(rng_seed)
        self.np_rng = np.random.default_rng(rng_seed)

        self.path = path

        # Width and height of the models input image
        self.wh = 64
        self.downsamples = 4
        self.downsamplesN = 16
        self.batch_size = batch_size
        self.filenames = sorted(
            [
                os.path.split(f)[1].split(".")[-2]
                for f in glob(os.path.join(self.path, "*.png"))
            ]
        )

        self.nChannels = 3

    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        actual_bs = min(self.batch_size, len(self.filenames) - idx * self.batch_size)
        grid = np.zeros(
            (actual_bs, self.downsamplesN, self.downsamplesN, 8), dtype=np.float32
        )
        grid[..., 0] = -np.inf  # o
        grid[..., 1] = 0.5  # x
        grid[..., 2] = 0.5  # y

        X = np.zeros((actual_bs, self.wh, self.wh, self.nChannels))

        for b in range(actual_bs):
            ellipse_data = np.load(
                os.path.join(
                    self.path, self.filenames[idx * self.batch_size + b] + ".npy"
                )
            )
            X[b] = skimage.io.imread(
                os.path.join(
                    self.path, self.filenames[idx * self.batch_size + b] + ".png"
                )
            )

            if len(ellipse_data) > 0:
                contained_mask = (
                    (ellipse_data[:, 0] > 0)
                    & (ellipse_data[:, 0] < self.wh)
                    & (ellipse_data[:, 1] > 0)
                    & (ellipse_data[:, 1] < self.wh)
                )

                # Mask grid cells whose center is within an ellipse so they can
                # be ignored in objectness loss
                cell_x, cell_y = np.meshgrid(
                    np.arange(0, self.wh, self.downsamples) + self.downsamples / 2,
                    np.arange(0, self.wh, self.downsamples) + self.downsamples / 2,
                )
                for ed in ellipse_data:
                    grid[b, ..., 7] += utils.points_in_ellipse(
                        cell_x, cell_y, *ed[:4], np.deg2rad(ed[4])
                    )

                grid[b, ..., 7] = grid[b, ..., 7] > 0

                ellipse_data = ellipse_data[contained_mask]

                for i, ed in enumerate(ellipse_data):
                    if ed.shape[0] == 0:
                        continue

                    pIdX = int(ed[0] // self.downsamples)
                    pIdY = int(ed[1] // self.downsamples)

                    grid[b, pIdY, pIdX, 0] = np.inf
                    grid[b, pIdY, pIdX, 1] = (
                        ed[0] - pIdX * self.downsamples
                    ) / self.downsamples
                    grid[b, pIdY, pIdX, 2] = (
                        ed[1] - pIdY * self.downsamples
                    ) / self.downsamples
                    grid[b, pIdY, pIdX, 3] = (
                        min(ed[2], ed[3]) / self.downsamples
                    )  # width
                    grid[b, pIdY, pIdX, 4] = (
                        max(ed[2], ed[3]) / self.downsamples
                    )  # height
                    a = ((ed[4]) % 180) if ed[2] < ed[3] else ((ed[4] - 90) % 180)
                    a = a / 180
                    grid[b, pIdY, pIdX, 5] = a
                    if ed[5] == 1:
                        grid[b, pIdY, pIdX, 6] = np.inf
                    else:
                        grid[b, pIdY, pIdX, 6] = -np.inf

        return tf.convert_to_tensor(X, dtype=tf.float32), grid
