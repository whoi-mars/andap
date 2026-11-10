import numpy as np
import os
import random
import shutil
from glob import glob
import skimage
import PIL.Image
import geopandas
import rasterio
import warnings

from . import utils
from .utils import posture_map


PIL.Image.MAX_IMAGE_PIXELS = None


def create_crops(ds_path, image_path, annotation_path, scale=1, counter=0):
    """
    Divides annotated regions of an image into 128px x 128px training samples.

    Args:
        ds_path (str): Path to where the crops will be stored.
        image_path (str): Path to the input image.
        annotation_path (str): Path to annotation files. These need to be called regions.gpkg and ellipses.gpkg.
        scale (float, optional): Factor by which the image size will be scaled. Defaults to 1.
        counter (int, optional): Crop counter used for file names.

    Returns:
        int: Value of crop counter after processing the image
    """
    crop_size = 128

    initial_counter = counter
    standing_counter = 0
    lying_counter = 0

    with rasterio.open(image_path) as img:
        crs = img.crs
        img_transform = img.transform

    assert crs.linear_units_factor[1] == 1.0, (
        "CRS must be projected and using units of meters."
    )

    if abs(img_transform.a) != abs(img_transform.e):
        warnings.warn(
            "X and Y pixel size of GeoTIFF differ by factor of "
            f"{img_transform.a / img_transform.e}. This will not be properly "
            "accounted for."
        )
    img_scale = abs(img_transform.a)

    regions_gdf = geopandas.read_file(
        os.path.join(annotation_path, "regions.gpkg")
    ).to_crs(crs)
    regions = regions_gdf.affine_transform((~img_transform).to_shapely()).geometry

    points, df = utils.load_points(os.path.join(annotation_path, "ellipses.gpkg"), crs)
    points = utils.apply_affine(~img_transform, np.asarray(points, dtype="float64"))
    ellipses = np.stack(
        [
            points[:, 0],
            points[:, 1],
            df["width [m]"],
            df["height [m]"],
            df["angle [deg]"],
            df["posture"],
        ],
        axis=1,
    )
    ellipses[:, :2] *= scale
    ellipses[:, 2:-2] *= scale / img_scale

    image = np.asarray(PIL.Image.open(image_path).convert("RGB"))

    for r in regions:
        rx_min, ry_min, rx_max, ry_max = np.int32(np.round(np.array(r.bounds)))
        im = image[ry_min:ry_max, rx_min:rx_max]
        if scale != 1:
            im = skimage.util.img_as_ubyte(
                skimage.transform.rescale(
                    im, [scale, scale], anti_aliasing=True, channel_axis=2
                )
            )
        rx = rx_min * scale
        ry = ry_min * scale
        rw = im.shape[1]
        rh = im.shape[0]
        xcount = rw // crop_size
        ycount = rh // crop_size
        for x in range(0, xcount * crop_size, crop_size):
            for y in range(0, ycount * crop_size, crop_size):
                skimage.io.imsave(
                    os.path.join(ds_path, f"{counter}.png"),
                    im[y : y + crop_size, x : x + crop_size],
                    check_contrast=False,
                )

                # Also include some ellipses from neighboring crops, which can
                # be used to ignore noobj grid cells within a GT bounding box
                # in loss function
                pad = 12
                contained_ellipses = ellipses[
                    (ellipses[:, 0] > rx + x - pad)
                    & (ellipses[:, 0] < rx + x + crop_size + pad)
                    & (ellipses[:, 1] > ry + y - pad)
                    & (ellipses[:, 1] < ry + y + crop_size + pad)
                ]
                contained_ellipses[:, 0] -= x + rx
                contained_ellipses[:, 1] -= y + ry
                with open(os.path.join(ds_path, f"{counter}.npy"), "wb") as f:
                    np.save(f, contained_ellipses)

                counter += 1

                # Posture of penguins contained in the crop without padding
                # (only used for counting)
                posture_contained = ellipses[
                    (ellipses[:, 0] > rx + x)
                    & (ellipses[:, 0] < rx + x + crop_size)
                    & (ellipses[:, 1] > ry + y)
                    & (ellipses[:, 1] < ry + y + crop_size),
                    -1,
                ]
                standing_counter += np.sum(posture_contained == posture_map["standing"])
                lying_counter += np.sum(posture_contained == posture_map["lying"])

    print(
        f"{image_path}: {standing_counter + lying_counter} penguins "
        f"({standing_counter} standing, {lying_counter} lying) on "
        f"{counter - initial_counter} samples"
    )
    return counter


def split_training_validation(ds_path, ratio, rng_seed=0):
    """
    Randomly splits samples in ds_path into a training dataset (ds_path/training/) and validation dataset (ds_path/validation/)

    Args:
        ds_path (str): Path to dataset.
        ratio (float): Fraction of samples that will be used for validation.
        rng_seed (int, optional): Seed for random number generator. Defaults to 0.
    """
    os.mkdir(os.path.join(ds_path, "training"))
    os.mkdir(os.path.join(ds_path, "validation"))
    filenames = sorted(
        [
            os.path.split(f)[1].split(".")[-2]
            for f in glob(os.path.join(ds_path, "*.png"))
        ]
    )
    rng = random.Random(rng_seed)
    rng.shuffle(filenames)
    split_index = round(len(filenames) * (1 - ratio))
    for f in filenames[:split_index]:
        shutil.move(
            os.path.join(ds_path, f + ".png"), os.path.join(ds_path, "training")
        )
        shutil.move(
            os.path.join(ds_path, f + ".npy"), os.path.join(ds_path, "training")
        )

    for f in filenames[split_index:]:
        shutil.move(
            os.path.join(ds_path, f + ".png"), os.path.join(ds_path, "validation")
        )
        shutil.move(
            os.path.join(ds_path, f + ".npy"), os.path.join(ds_path, "validation")
        )


def crop_validation(ds_path):
    """
    Splits 128x128 samples into four 64x64 samples used for validation.

    Args:
        ds_path (str): Path to dataset.
    """
    path = os.path.join(ds_path, "validation")
    filenames = sorted(
        [os.path.split(f)[1].split(".")[-2] for f in glob(os.path.join(path, "*.png"))]
    )

    # Also save ellipses +- pad pixels outside crop (for inside_mask in data generator)
    pad = 12

    for f in filenames:
        if "_" in f:
            continue
        im = skimage.io.imread(os.path.join(path, f + ".png"))
        ellipse_data = np.load(os.path.join(path, f + ".npy"))
        contained_mask = []

        # x<64, y<64
        skimage.io.imsave(
            os.path.join(path, f + "_0" + ".png"), im[:64, :64], check_contrast=False
        )
        if len(ellipse_data) > 0:
            contained_mask = (
                (ellipse_data[:, 0] > -pad)
                & (ellipse_data[:, 0] < 64 + pad)
                & (ellipse_data[:, 1] > -pad)
                & (ellipse_data[:, 1] < 64 + pad)
            )
        with open(os.path.join(path, f + "_0" + ".npy"), "wb") as npyf:
            np.save(npyf, ellipse_data[contained_mask])

        # x>64, y<64
        skimage.io.imsave(
            os.path.join(path, f + "_1" + ".png"), im[:64, 64:], check_contrast=False
        )
        if len(ellipse_data) > 0:
            contained_mask = (
                (ellipse_data[:, 0] > 64 - pad)
                & (ellipse_data[:, 0] < 128 + pad)
                & (ellipse_data[:, 1] > -pad)
                & (ellipse_data[:, 1] < 64 + pad)
            )
            ed = ellipse_data[contained_mask]
            ed[:, 0] -= 64
        else:
            ed = []
        with open(os.path.join(path, f + "_1" + ".npy"), "wb") as npyf:
            np.save(npyf, ed)

        # x<64, y>64
        skimage.io.imsave(
            os.path.join(path, f + "_2" + ".png"), im[64:, :64], check_contrast=False
        )
        if len(ellipse_data) > 0:
            contained_mask = (
                (ellipse_data[:, 0] > -pad)
                & (ellipse_data[:, 0] < 64 + pad)
                & (ellipse_data[:, 1] > 64 - pad)
                & (ellipse_data[:, 1] < 128 + pad)
            )
            ed = ellipse_data[contained_mask]
            ed[:, 1] -= 64
        else:
            ed = []
        with open(os.path.join(path, f + "_2" + ".npy"), "wb") as npyf:
            np.save(npyf, ed)

        # x>64, y>64
        skimage.io.imsave(
            os.path.join(path, f + "_3" + ".png"), im[64:, 64:], check_contrast=False
        )
        if len(ellipse_data) > 0:
            contained_mask = (
                (ellipse_data[:, 0] > 64 - pad)
                & (ellipse_data[:, 0] < 128 + pad)
                & (ellipse_data[:, 1] > 64 - pad)
                & (ellipse_data[:, 1] < 128 + pad)
            )
            ed = ellipse_data[contained_mask]
            ed[:, 1] -= 64
            ed[:, 0] -= 64
        else:
            ed = []
        with open(os.path.join(path, f + "_3" + ".npy"), "wb") as npyf:
            np.save(npyf, ed)

        os.remove(os.path.join(path, f + ".png"))
        os.remove(os.path.join(path, f + ".npy"))


def prepare_training_data(
    ds_path, image_paths, annotation_paths, scale=1.0, validation_split_ratio=0.15
):
    """
    Create training and validation dataset from set of annotated images.

    Args:
        ds_path (str): Path where dataset will be stored.
        image_paths (list[str]): Paths to images.
        annotation_paths (list[str]): Paths to directories containing annotation files. These need to be called regions.gpkg and ellipses.gpkg.
        scale (float or list[float], optional): Factors by which the images will be scaled. If scalar then the same scaling will be applied to all images. Defaults to 1.
        validation_split_ratio (float, optional): Fraction of samples that will be used for validation. Defaults to 0.15.
    """
    if np.isscalar(scale):
        scale = [scale] * len(image_paths)

    os.makedirs(ds_path, exist_ok=True)

    counter = 0
    for ip, ap, s in zip(image_paths, annotation_paths, scale):
        counter = create_crops(ds_path, ip, ap, s, counter)
    split_training_validation(ds_path, validation_split_ratio)
    crop_validation(ds_path)
