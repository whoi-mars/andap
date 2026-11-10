import numpy as np
import shapely
from tqdm import tqdm
import itertools
import scipy
import rasterio
import geopandas

from . import model
from . import utils


def get_precision_recall_confusion(
    ground_truth_files,
    prediction_files,
    region_files,
    image_files,
    objectness_thresholds,
    overlap_thresholds,
    class_thresholds,
    assignment_overlap_thresh=0.4,
):
    """
    Computes precision and recall for the object detection and confusion matrix for the posture classification on an annotated test dataset.

    For precision and recall it only matters how many of the predicted ellipses match with a ground truth penguin, posture is not considered.
    For the posture classification confusion matrix, only true positives from object detection are considered.
    Args:
        ground_truth_files (list[str]): Names of GeoPackage files containing ground truth annotations in the same format as produced by colony.run_detection().
        prediction_files (list[str]): Filenames of the corresponding outputs from the automated object detection. These must be created with colony.run_detection() with the save_model_output arguemnt set to True.
        region_files (list[str]): Names of GeoPackage files containing polygons drawn around the annotated regions. Everything outside will be ignored.
        image_files (list[str]): List of GeoTIFFs associated with the other files, used to get coordinate transformations.
        objectness_thresholds (ndarray): 1D-array of objectness thresholds to be tested.
        overlap_thresholds (ndarray): 1D-array of overlap thresholds for non-max suppression to be tested.
        class_thresholds (ndarray): 1D-array of classification thresholds to be tested.
        assignment_overlap_thresh (float, optional): Threshold for KLD overlap from which on two ellipses are counted as covering the same object. Defaults to 0.4.

    Returns:
        precision (ndarray): Precision of object detection. First index iterates objectness_thresholds, second index is for overlap_thresholds.
        recall (ndarray): Recall, same shape as precision.
        tp (ndarray): Number of true positives of object detection. Same shape as precision.
        fp (ndarray): Number of false positives of object detection. Same shape as precision.
        fn (ndarray): Number of false negatives of object detection. Same shape as precision.
        confusion_matrix (ndarray): Confusion matrix for posture classification. The first three indices iterate thresholds for objectness, overlap and classification in that order. Last two indices distinguish posture of ground truth and prediction (index=0: standing, index=1: lying)
    """
    # Number of true positives
    tp = np.zeros((len(objectness_thresholds), len(overlap_thresholds)), dtype="int")
    # Number of false positives
    fp = np.zeros((len(objectness_thresholds), len(overlap_thresholds)), dtype="int")
    # Number of false negatives
    fn = np.zeros((len(objectness_thresholds), len(overlap_thresholds)), dtype="int")

    # Counts for posture classification task
    confusion_matrix = np.zeros(
        (
            len(objectness_thresholds),
            len(overlap_thresholds),
            len(class_thresholds),
            2,
            2,
        ),
        dtype="int",
    )

    # Get predictions and ground truth for each region
    pred = []
    true = []
    for gtf, pf, rf, imf in zip(
        ground_truth_files, prediction_files, region_files, image_files
    ):
        with rasterio.open(imf) as img:
            crs = img.crs
            img_origin = (img.transform.c, img.transform.f)
        assert crs.linear_units_factor[1] == 1.0, (
            "CRS must be projected and using units of meters."
        )

        regions_gdf = geopandas.read_file(rf).to_crs(crs)
        regions = regions_gdf.geometry

        # Subtract image origin from coordinates to avoid numerical issues
        subtract_origin = lambda x, y: (x - img_origin[0], img_origin[1] - y)
        so_series = lambda geom: shapely.ops.transform(subtract_origin, geom)
        regions = regions.apply(so_series)

        pred_points, pred_gdf = utils.load_points(pf, crs)
        ellipses_pred = np.stack(
            [
                pred_gdf["objectness"],
                pred_points[:, 0] - img_origin[0],
                img_origin[1] - pred_points[:, 1],
                pred_gdf["width [m]"],
                pred_gdf["height [m]"],
                pred_gdf["angle [deg]"] * np.pi / 180,
                pred_gdf["posture"],
            ],
            axis=1,
        )

        gt_points, gt_gdf = utils.load_points(gtf, crs)
        ellipses_gt = np.stack(
            [
                np.ones_like(gt_points[:, 0]),
                gt_points[:, 0] - img_origin[0],
                img_origin[1] - gt_points[:, 1],
                gt_gdf["width [m]"],
                gt_gdf["height [m]"],
                gt_gdf["angle [deg]"] * np.pi / 180,
                gt_gdf["posture"],
            ],
            axis=1,
        )

        for reg in regions:
            pred_contained_mask = shapely.intersects_xy(reg, ellipses_pred[:, 1:3])
            # Only predictions with objectness higher than minimum in range matter
            pred_min_objectness_mask = ellipses_pred[:, 0] >= np.min(
                objectness_thresholds
            )
            pred.append(
                np.array(ellipses_pred[pred_contained_mask & pred_min_objectness_mask])
            )

            gt_contained_mask = shapely.intersects_xy(reg, ellipses_gt[:, 1:3])
            true.append(np.array(ellipses_gt[gt_contained_mask]))

    # Get for each of the regions overlap matrix
    max_overlap = []
    for i in range(len(pred)):
        ## Non-max suppression for prediction
        # The overlap matrix quickly gets huge and contains mostly entries of
        # ellipses far away. Therefore we use a different approach here to get
        # max_overlap, where only overlap with ellipses closer than 3m is calculated.
        distance_thresh = 3
        objectness = pred[i][:, 0]
        max_overlap.append(np.zeros_like(objectness))
        for j in tqdm(range(len(pred[i]))):
            nearby_ellipse_indices = np.where(
                np.linalg.norm(pred[i][:, 1:3] - pred[i][j, 1:3], axis=1)
                < distance_thresh
            )[0]
            nearby_ellipse_indices = nearby_ellipse_indices[nearby_ellipse_indices != j]
            if len(nearby_ellipse_indices) > 0:
                kld = model.kld(pred[i][j], pred[i][nearby_ellipse_indices])
                overlap = 1 / (1 + kld)
                mask = objectness[j] < objectness[nearby_ellipse_indices]
                max_overlap[i][j] = np.max(np.where(mask, overlap, 0))

    for i, j in tqdm(
        itertools.product(
            range(len(objectness_thresholds)), range(len(overlap_thresholds))
        ),
        total=len(objectness_thresholds) * len(overlap_thresholds),
    ):
        obj_thresh = objectness_thresholds[i]
        overlap_thresh = overlap_thresholds[j]

        for r in range(len(pred)):
            # Only take predicted objects with high enough objectness that are not
            # overlapped by higher objectness ellipses
            pred_mask = (max_overlap[r] < overlap_thresh) & (pred[r][:, 0] > obj_thresh)
            if not (np.any(pred_mask) or len(true[r]) > 0):
                continue

            # Predicted and ground truth ellipses for region r
            ps = pred[r][pred_mask]
            ts = true[r]

            # Assign predicted to ground truth ellipses to maximise overlap
            # between matched pairs
            xx, yy = np.meshgrid(np.arange(len(ps)), np.arange(len(ts)))
            overlap_true_pred = 1 / (1 + model.kld(ts[yy], ps[xx]))
            true_idx, pred_idx = scipy.optimize.linear_sum_assignment(
                overlap_true_pred, maximize=True
            )
            matched_true_idx = true_idx[
                overlap_true_pred[true_idx, pred_idx] > assignment_overlap_thresh
            ]
            matched_pred_idx = pred_idx[
                overlap_true_pred[true_idx, pred_idx] > assignment_overlap_thresh
            ]

            tp_ = len(matched_true_idx)
            fp_ = len(ps) - tp_
            fn_ = len(ts) - tp_
            tp[i, j] += tp_
            fp[i, j] += fp_
            fn[i, j] += fn_

            for k in range(len(class_thresholds)):
                true_class = ts[matched_true_idx, 6] > 0.5
                pred_class = ps[matched_pred_idx, 6] > class_thresholds[k]

                confusion_matrix[i, j, k, 0, 0] += np.sum((~true_class & ~pred_class))
                confusion_matrix[i, j, k, 0, 1] += np.sum((~true_class & pred_class))
                confusion_matrix[i, j, k, 1, 0] += np.sum((true_class & ~pred_class))
                confusion_matrix[i, j, k, 1, 1] += np.sum((true_class & pred_class))

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)

    return prec, rec, tp, fp, fn, confusion_matrix


def f1_acc(
    ground_truth_files,
    prediction_files,
    region_files,
    image_files,
    objectness_thresholds=None,
    overlap_thresholds=None,
    class_thresholds=None,
    assignment_overlap_thresh=0.4,
):
    """
    Get F1-score for object detection and accuracy for posture classification on annotated test dataset, using a range of objectness, overlap and class thresholds.

    Args:
        ground_truth_files (list[str]): Names of GeoPackage files containing ground truth annotations in the same format as produced by colony.run_detection().
        prediction_files (list[str]): Filenames of the corresponding outputs from the automated object detection. These must be created with colony.run_detection() with the save_model_output arguemnt set to True.
        region_files (list[str]): Names of GeoPackage files containing polygons drawn around the annotated regions. Everything outside will be ignored.
        image_files (list[str]): List of GeoTIFFs associated with the other files, used to get coordinate transformations.
        objectness_thresholds (ndarray, optional): 1D-array of objectness thresholds to be tested. If None (default), this will be set to np.linspace(0,1,51).
        overlap_thresholds (ndarray, optional): 1D-array of overlap thresholds to be tested. If None (default), this will be set to np.linspace(0,1,51).
        class_thresholds (ndarray, optional): 1D-array of class thresholds to be tested. If None (default), this will be set to np.linspace(0,1,51).
        assignment_overlap_thresh (float, optional): Threshold for KLD overlap from which on two ellipses are counted as covering the same object. Defaults to 0.4.

    Returns:
        f1 (ndarray): Object detection F1 score for the different thresholds. First and second axis run over objectness and overlap thresholds, respectively.
        accuracy (ndarray): Posture classification accuracy for different classification thresholds, evaluated at optimal objectness and overlap thresholds.
    """

    if objectness_thresholds is None:
        objectness_thresholds = np.linspace(0, 1, 51)
    else:
        objectness_thresholds = np.asarray(objectness_thresholds)

    if overlap_thresholds is None:
        overlap_thresholds = np.linspace(0, 1, 51)
    else:
        overlap_thresholds = np.asarray(overlap_thresholds)

    if class_thresholds is None:
        class_thresholds = np.linspace(0, 1, 51)
    else:
        class_thresholds = np.asarray(class_thresholds)

    prec, rec, tp, fp, fn, class_confusion = get_precision_recall_confusion(
        ground_truth_files,
        prediction_files,
        region_files,
        image_files,
        objectness_thresholds,
        overlap_thresholds,
        class_thresholds,
        assignment_overlap_thresh,
    )
    f1 = 2 / (1 / rec + 1 / prec)

    obj_thresh_idx, overlap_thresh_idx = np.unravel_index(np.nanargmax(f1), f1.shape)
    print(
        f"Best F1-score {f1[obj_thresh_idx, overlap_thresh_idx]:.4f} "
        f"({tp[obj_thresh_idx, overlap_thresh_idx]} true positives, "
        f"{fp[obj_thresh_idx, overlap_thresh_idx]} false positives and "
        f"{fn[obj_thresh_idx, overlap_thresh_idx]} false negatives) "
        f"reached with obj_threshold={objectness_thresholds[obj_thresh_idx]} and "
        f"overlap_threshold={overlap_thresholds[overlap_thresh_idx]}."
    )

    cf = class_confusion[obj_thresh_idx, overlap_thresh_idx]
    accuracy = (cf[:, 0, 0] + cf[:, 1, 1]) / np.sum(cf, axis=(1, 2))
    class_thresh_idx = np.unravel_index(np.nanargmax(accuracy), accuracy.shape)
    print(
        f"Best classification accuracy {accuracy[class_thresh_idx]:.4f} "
        f"reached with class_threshold={class_thresholds[class_thresh_idx]}."
    )

    return f1, accuracy
