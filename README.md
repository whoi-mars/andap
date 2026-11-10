# ANDAP: Automatic Nest Detection for Adélie Penguins

A Python package for automatically detecting Adélie penguins on aerial images and distinguishing nests from non-nesting penguins.

## Installation
The ANDAP package can be installed with pip from the GitHub repository:
```bash
python -m pip install 'andap@git+https://github.com/whoi-mars/andap.git@main'
```

## Getting Started
The examples below demonstrate how ANDAP can be used in practice.
To understand these, you already need to be familiar with the different steps of the nest counting analysis, which are explained in the accompanying paper (in preparation).
The readme is only meant to serve as a starting point, for more details on advanced options please consult docstrings in the source code.

### Running the full analysis using a pretrained model
To do any kind of analysis, you first need to create a `Colony` object, specifying a name, a path where you want to store the results and the path to your orthomosaic in the GeoTIFF format.
```python
import andap

col = andap.colony.Colony("platter", "./results/platter/", "./orthomsaics/platter.tiff")
```
To detect penguins on the image, you also need to initialise the neural network by creating a `Model` object.
Here we want to use weights trained on the Danger Islands data with a 3.2 cm resolution.
```python
mod = andap.model.Model(weights="3.2cm_DI")
```
For more information on available models, see the `andap.model.pretrained_models` dictionary.

The penguin detection can now be started with
```python
col.run_detection(mod)
```
If you want to restrict the object detection to only some parts of the full image, for example to exclude ice covered areas, you can use the `restrict_regions` argument of `run_detection()` to supply either a GeoPackage file or a shapely geometry masking the areas to be processed.
`run_detection()` and all following steps store the intermediate results as GeoPackage files in the results directory specified while initialising the `Colony` object.
Repeated execution of the same analysis steps will overwrite previous results, unless you specify different filenames.

Delineating subcolonies using alpha shapes and removing penguins outside subcolonies is done by calling the following two methods:
```python
col.delineate_subcolonies(alpha_radius=0.75)
col.filter_outside_subcolonies()
```

Non-nesting penguins within the colony are removed from the nest candidates in the pair removal step:
```python
col.run_pair_removal()
```
Without arguments, this will automatically determine distance thresholds to split the distributions of mutual nearest-neighbor distances as explained in the paper.
You can also set them manually using the `threshold_dists` argument.
To check how well the algorithm for finding the thresholds works, you can visualize them along with histograms of the nearest-neighbor distances by calling `run_pair_removal(plot=True)` (this requires an interactive matplotlib backend).

After these steps, the results directory should now contain a `platter_nests.gpkg` file, containing the positions of all identified nests.

### Combined analysis of multiple colonies
If you have surveyed several smaller colonies, it might make sense analyze them together, as the algorithm for determining the pair distance thresholds works better with more data.
The disadvantage is that this disregards any effect that different local terrain can have on inter-nest distances.

To run a combined analysis, you first need to initialize the different `Colony` objects (e.g. `col1`, `col2`, `col3`) and the object detection model `mod` as before.
Then you can use the following code:
```python
import os
import scipy

cols = [col1, col2, col3]

ll_dists = []
ss_dists = []
ls_dists = []

for col in cols:
    col.run_detection(mod)
    col.delineate_subcolonies()
    col.filter_outside_subcolonies()

    penguins = andap.utils.load_penguins(
        os.path.join(col.results_dir, col.colony_penguins_fname), col.crs
    )

    tree = scipy.spatial.cKDTree(penguins[:, :2])
    distances, neighbors = tree.query(penguins[:, :2], k=2)
    distances = distances[:, 1]
    neighbors = neighbors[:, 1]

    ll_d, ss_d, ls_d = andap.pair_detection.get_distances_by_type(
        penguins, distances, neighbors
    )
    ll_dists.extend(ll_d)
    ss_dists.extend(ss_d)
    ls_dists.extend(ls_d)

threshold_dists = (
    andap.pair_detection.find_threshold(ll_dists, plot=True, title="Lying-Lying"),
    andap.pair_detection.find_threshold(ss_dists, plot=True, title="Standing-Standing"),
    andap.pair_detection.find_threshold(ls_dists, plot=True, title="Lying-Standing"),
)

for col in cols:
    col.run_pair_removal(threshold_dists=threshold_dists)
```

### Training your own object detection model
The available weights for the object detection model have all been trained on a survey of the Danger Islands.
To analyse colonies in very different locations, or to work with image resolutions not covered by the available weights, it might be useful to train your own model.

#### Preparing training data
To train the object detection model, you first need training data.
We recommend as an annotation tool the [ClickPoints software](https://clickpoints.readthedocs.io), which allows to draw ellipses and encode posture information using different marker types.
ANDAP requires annotations to come in two GeoPackage files:
`ellipses.gpkg` contains markers on the center of each penguin, with additional fields for width, height and angle of elliptical bounding boxes around them and another field indicating penguin posture.
`regions.gpkg` contains polygons that delimit the annotated regions of the image.
To convert a ClickPoints database to this format, you can use the following code snippet.
This assumes that you have marked the annotated regions using rectangles with the type name `region` and have used two types of elliptical markers with the names `standing` and `lying` for the penguins:
```python
import clickpoints
import andap
import os
import numpy as np
import rasterio
import shapely
import pandas
import geopandas


def clickpoints_to_gpkg(clickpoints_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with clickpoints.DataFile(clickpoints_file) as db:
        image = db.getImages()[0]
        img_path = os.path.join(os.path.dirname(clickpoints_file), image.filename)
        with rasterio.open(img_path) as img:
            crs = img.crs
            img_transform = img.transform

        regions_cdb = db.getRectangles(image, type="region")
        ellipses_cdb = db.getEllipses(image)

    ellipses = np.zeros((len(ellipses_cdb), 6))
    for i, e in enumerate(ellipses_cdb):
        ellipses[i] = [
            e.x,
            e.y,
            e.width,
            e.height,
            e.angle,
            andap.utils.posture_map[e.type.name],
        ]

    coords = andap.utils.apply_affine(
        img_transform, np.asarray(ellipses[:, :2], dtype="float64")
    )
    ellipses[:, 2:4] *= abs(img.transform.a)
    df = pandas.DataFrame(
        ellipses[:, 2:], columns=["width [m]", "height [m]", "angle [deg]", "posture"]
    )
    andap.utils.save_points(os.path.join(output_dir, "ellipses.gpkg"), crs, coords, df)

    regions = []
    for r in regions_cdb:
        bounds = [r.x, r.y, r.x + r.width, r.y + r.height]
        regions.append(
            shapely.affinity.affine_transform(
                shapely.box(*bounds), (img_transform).to_shapely()
            )
        )

    gdf = geopandas.GeoDataFrame({"geometry": regions}, crs=crs)
    gdf.to_file(os.path.join(output_dir, "regions.gpkg"), driver="GPKG")


clickpoints_to_gpkg("path/to/annotations.cdb", "./annotations/")
```

Once the annotations are in the right format, they can be passed to ANDAP, which will divide the annotated regions into small crops and then split them up into a training and a validation dataset.
At this point, you can also rescale the images if you want to train an object detection model for a different image resolution.
```python
andap.data_preparation.prepare_training_data(
    "./training_data/",
    ["/path/to/image_A.tiff", "/path/to/image_B.tiff"],
    ["/path/to/annotations_A/", "/path/to/annotations_B/"],
    scale=3.2 / 4.3,  # Downscale from 3.2cm to 4.3cm resolution
)
```

#### Training the model
The following code first creates a new `Model` and two DataGenerators, which will supply the model with training and validation samples.
The model is then trained for 200 epochs, saving the weights which perform best on the validation dataset to `./models/model.weights.h5`.
Afterwards, it plots the loss curves showing how the model's performance improves.
```python
import numpy as np
import matplotlib.pyplot as plt

dg_val = andap.data_generator.DataGen("./training_data/validation")
dg_train = andap.data_generator.DataGenAugment("./training_data/training")

n_epochs = 200

mod = andap.model.Model()
history = mod.train(dg_train, dg_val, "./models/", epochs=n_epochs)

epochs = np.arange(1, n_epochs + 1)
for k in history.history.keys():
    if "val" in k:
        plt.plot(epochs, history.history[k], label=k, linestyle="--")
    else:
        plt.plot(epochs, history.history[k], label=k)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
```
The model is simultaneously trained for different tasks:
Distinguishing if there is a penguin at all, placing an elliptical bounding box around it, and classifying its posture.
Each of these tasks has its own loss term contributing to the total optimization objective, weighted by the parameters `object_loss_scale`, `kld_loss_scale` and `class_loss_scale`, which can be passed to `train()`.
To counteract overfitting in one of the tasks (indicated by an increase of validation loss for that task while the training loss keeps decreasing) while the model is still improving on the other tasks, you need to reduce the weight parameter for the overfitting task (or increase those for the other tasks) and restart training.

#### Determining decision thresholds
To actually use the trained model for penguin detection, the objectness, overlap and class thresholds need to be set appropriately for the non-maximum suppression and posture classification.
To choose these thresholds to maximize the F1-score for the object detection and the accuracy for the posture classification on the validation data, simply run
```python
mod.f1_acc_thresholding(dg_val)
```

#### Evaluating model performance
After training the model and setting the thresholds, it is recommended to check its performance on additional annotated test data different from the training and validation datasets that were used to tune the parameters.

To do this, you first need to run the object detection on annotated images.
Since you might later want to evaluate how well the model performs on the test data for different thresholds, it is necessary to tell `run_detection()` to save the full output of the neural network, without applying non-max suppression and the class threshold.
```python
import os

test_data_paths = ["./annotations_A/", "./annotations_B/"]
cols = [col_A, col_B]

for col, dp in zip(cols, test_data_paths):
    col.run_detection(
        mod, restrict_regions=os.path.join(dp, "regions.gpkg"), save_model_output=True
    )
```
To get the F1-score for the object detection and the accuracy of the posture classification using the thresholds determined earlier, run
```python
gt_files = [os.path.join(dp, "ellipses.gpkg") for dp in test_data_paths]
pred_files = [
    os.path.join(col.results_dir, "model_output_" + col.penguins_fname) for col in cols
]
region_files = [os.path.join(dp, "regions.gpkg") for dp in test_data_paths]
image_files = [col.image_path for col in cols]

f1, acc = andap.evaluation.f1_acc(
    gt_files,
    pred_files,
    region_files,
    image_files,
    objectness_thresholds=[mod.obj_thresh],
    overlap_thresholds=[mod.overlap_thresh],
    class_thresholds=[mod.class_thresh],
)
```

If you want to see how the performance varies with different thresholds, you can plot the precision-recall curve for the object detection and the ROC curve for the posture classification:
```python
import numpy as np
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2))

# Vary only objectness thresholds for precision/recall
prec, rec, *_ = andap.evaluation.get_precision_recall_confusion(
    gt_files,
    pred_files,
    region_files,
    image_files,
    objectness_thresholds=np.linspace(0, 1, 51),
    overlap_thresholds=[mod.overlap_thresh],
    class_thresholds=[mod.class_thresh],
)

ax1.plot(rec[:, 0], prec[:, 0])
ax1.set_xlabel("Recall")
ax1.set_ylabel("Precision")
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_aspect("equal")

# Vary only class thresholds for posture classification confusion matrix
*_, confusion = andap.evaluation.get_precision_recall_confusion(
    gt_files,
    pred_files,
    region_files,
    image_files,
    objectness_thresholds=[mod.obj_thresh],
    overlap_thresholds=[mod.overlap_thresh],
    class_thresholds=np.linspace(0, 1, 51),
)

ax2.plot(
    (confusion[0, 0, :, 1, 0]) / np.sum(confusion[0, 0, :, 1, :], axis=(1)),
    (confusion[0, 0, :, 0, 0]) / np.sum(confusion[0, 0, :, 0, :], axis=(1)),
)
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_aspect("equal")

plt.plot()
```
