import tensorflow as tf
import numpy as np
import os
import scipy
from tqdm import tqdm
import itertools

try:
    from importlib import resources
except ImportError:
    import importlib_resources as resources


pretrained_models = {
    "3.2cm_DI": {
        "resolution": 3.2,  # cm/px
        "objectness_threshold": 0.26,
        "overlap_threshold": 0.43,
        "classification_threshold": 0.40,
        "training_params": {
            "epochs": 200,
            "object_loss_scale": 37.0,
            "kld_loss_scale": 1.0,
            "class_loss_scale": 0.011,
        },
    },
    "4.3cm_DI": {
        "resolution": 4.3,
        "objectness_threshold": 0.34,
        "overlap_threshold": 0.44,
        "classification_threshold": 0.36,
        "training_params": {
            "epochs": 200,
            "object_loss_scale": 31.0,
            "kld_loss_scale": 1.0,
            "class_loss_scale": 0.007,
        },
    },
    "5.4cm_DI": {
        "resolution": 5.4,
        "objectness_threshold": 0.29,
        "overlap_threshold": 0.46,
        "classification_threshold": 0.44,
        "training_params": {
            "epochs": 200,
            "object_loss_scale": 13.0,
            "kld_loss_scale": 1.0,
            "class_loss_scale": 9e-6,
        },
    },
}


def kld(A, B):
    """
    Calculates Kullback-Leibler divergence of Gaussian distributions characterized by ellipses A and B.

    If A and B have more than one dimension and contain multiple ellipses, they will be compared elementwise.
    For details of the formula used see doi:10.48550/arXiv.2106.01883
    Args:
        A (ndarray or tf.Tensor): First ellipse(s). Last dimension must follow model output index convention, angle must be in radians and x,y,w,h must have identical units.
        B (ndarray or tf.Tensor): Second ellipse(s).

    Returns:
        ndarray or tf.Tensor: KLD, shape is that of A and B without last dimension.
    """
    eps = tf.keras.backend.epsilon()

    dx = B[..., 1] - A[..., 1]
    dy = B[..., 2] - A[..., 2]
    da = B[..., 5] - A[..., 5]

    w_A = A[..., 3] + eps
    w_B = B[..., 3] + eps
    h_A = A[..., 4] + eps
    h_B = B[..., 4] + eps

    kld_ = 0.0
    kld_ += (
        4 * (dx * tf.math.cos(A[..., 5]) + dy * tf.math.sin(A[..., 5])) ** 2 / w_A**2
    )
    kld_ += (
        4 * (dy * tf.math.cos(A[..., 5]) - dx * tf.math.sin(A[..., 5])) ** 2 / h_A**2
    )
    kld_ += (h_B**2 / w_A**2 + w_B**2 / h_A**2) * tf.math.sin(da) ** 2
    kld_ += (h_B**2 / h_A**2 + w_B**2 / w_A**2) * tf.math.cos(da) ** 2
    kld_ += tf.math.log(h_A**2 / h_B**2) + tf.math.log(w_A**2 / w_B**2)
    kld_ *= 0.5
    kld_ -= 1
    if tf.is_tensor(A):
        return kld_
    else:
        return kld_.numpy()


def transform_output(output):
    """
    Turns the model output where o,a are logits, x,y are relative to the grid cell location, x,y,w,h are in units of grid cell size and a is in (0,1) into the more useful format where o,a are probabilities, x,y are pixel coordinates within the full sample, w,h are in units of pixels and a is in radians.

    Args:
        output (tf.Tensor): The model output tensor to be tranformed.

    Returns:
        tf.Tensor: Transformed version of output.
    """
    ds = 4
    xx, yy = tf.meshgrid(
        tf.range(0, 16, dtype="float32"), tf.range(0, 16, dtype="float32")
    )

    new_o = tf.sigmoid(output[..., 0])
    new_x = (output[..., 1] + xx) * ds
    new_y = (output[..., 2] + yy) * ds
    new_w = output[..., 3] * ds
    new_h = output[..., 4] * ds
    new_a = output[..., 5] * np.pi
    new_c = tf.sigmoid(output[..., 6])

    if output.shape[-1] == 7:
        return tf.stack((new_o, new_x, new_y, new_w, new_h, new_a, new_c), axis=-1)
    else:
        return tf.stack(
            (new_o, new_x, new_y, new_w, new_h, new_a, new_c, output[..., 7]), axis=-1
        )


class Loss(tf.keras.losses.Loss):
    def __init__(
        self,
        *args,
        object_scale,
        kld_scale,
        class_scale,
        batch_size,
        obj_alpha,
        obj_gamma,
        class_alpha,
        class_gamma,
        **kwargs,
    ):
        super(Loss, self).__init__(*args, name=type(self).__name__, **kwargs)
        self.object_scale = object_scale
        self.kld_scale = kld_scale
        self.class_scale = class_scale
        self.batch_size = batch_size

        self.obj_fce = tf.keras.losses.BinaryFocalCrossentropy(
            apply_class_balancing=True,
            alpha=obj_alpha,
            gamma=obj_gamma,
            from_logits=True,
            reduction=None,
        )
        self.class_fce = tf.keras.losses.BinaryFocalCrossentropy(
            apply_class_balancing=True,
            alpha=class_alpha,
            gamma=class_gamma,
            from_logits=True,
            reduction=None,
        )

    # Object / No Object Loss (focal loss)
    def obj_loss(self, y_true, y_pred):
        # Ignore grid cells covered by a GT penguin without containing its center in loss
        loss_mask = (y_true[..., 7] == 0) | (y_true[..., 0] > 0)

        return (
            tf.reduce_mean(
                self.obj_fce(
                    tf.sigmoid(y_true[..., 0][loss_mask]), y_pred[..., 0][loss_mask]
                )
            )
            * self.object_scale
        )

    # Kullback-Leibler Divergence Loss
    def kld_loss(self, y_true, y_pred):
        y_true = transform_output(y_true)
        y_pred = transform_output(y_pred)

        kld_ = kld(y_true[y_true[..., 0] > 0], y_pred[y_true[..., 0] > 0])
        L = 1 - 1 / (1 + tf.math.log(kld_ + 1))
        return tf.where(
            tf.math.reduce_any(y_true[..., 0] > 0),
            self.kld_scale * tf.reduce_mean(L),
            0.0,
        )

    # Posture Classification Loss (focal loss)
    def class_loss(self, y_true, y_pred):
        class_true = y_true[..., 6][y_true[..., 0] > 0]
        class_pred = y_pred[..., 6][y_true[..., 0] > 0]

        return (
            tf.where(
                tf.shape(class_true)[0] > 0,
                tf.reduce_mean(self.class_fce(tf.sigmoid(class_true), class_pred)),
                0.0,
            )
            * self.class_scale
        )

    def call(self, y_true, y_pred):
        return (
            self.obj_loss(y_true, y_pred)
            + self.kld_loss(y_true, y_pred)
            + self.class_loss(y_true, y_pred)
        )


class Model:
    def __init__(
        self,
        weights=None,
        obj_thresh=None,
        overlap_thresh=None,
        class_thresh=None,
        rng_seed=0,
    ):
        """
        Initializes penguin detection model.

        Args:
            weights (str, optional): Weights to initalize network with. Can be the name of a pretrained model, the path to a weights file or None (default). If None, no weights are loaded.
            obj_thresh (float, optional): Objectness threshold for non-max suppression. Defaults to None but will be set to the value from andap.model.pretrained_models if pretrained weights are loaded.
            overlap_thresh (float, optional): Overlap threshold for non-max suppression. Defaults to None but will be set to the value from andap.model.pretrained_models if pretrained weights are loaded.
            class_thresh (float, optional): Posture classification threshold. Defaults to None but will be set to the value from andap.model.pretrained_models if pretrained weights are loaded.
            rng_seed (int, optional): Sets the the tensorflow global random seed to ensure that weight initialization is reproducible. Defaults to 0.
        """
        self.obj_thresh = obj_thresh
        self.overlap_thresh = overlap_thresh
        self.class_thresh = class_thresh

        inp_wh = 64     # Width, height of input image
        nChannels = 3   # Number of channels of the input image
        ds = 4          # Downsampling: width/height of grid cells in pixels
        nFilters = 7    # Number of outputs per grid cell

        tf.keras.utils.set_random_seed(rng_seed)

        # 3x3 convolution with BatchNormalization
        def conv_block(x, filters, strides=1):
            x = tf.keras.layers.Conv2D(
                filters,
                3,
                strides=strides,
                activation="relu",
                padding="same",
                use_bias=False,
                kernel_initializer="he_normal",
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("relu")(x)
            return x

        # Residual block with two 3x3 convs and parallel identity connection
        def res_block(x, filters):
            fx = conv_block(x, filters)
            fx = tf.keras.layers.Conv2D(
                filters,
                3,
                activation=None,
                padding="same",
                use_bias=False,
                kernel_initializer="he_normal",
            )(fx)
            fx = tf.keras.layers.BatchNormalization()(fx)
            out = tf.keras.layers.Add()([x, fx])
            out = tf.keras.layers.Activation("relu")(out)
            return out

        inputs = tf.keras.Input((inp_wh, inp_wh, nChannels))

        # 64x64px layers
        x = conv_block(inputs, 32)
        x = res_block(x, 32)

        # 32x32
        x = conv_block(x, 64, strides=2)
        skip = conv_block(x, 64)

        # 16x16
        x = conv_block(skip, 128, strides=2)
        x = conv_block(x, 128)

        # Add skip connection from 32x32
        x = tf.keras.layers.Add()([x, conv_block(skip, 128, strides=2)])

        x = res_block(x, 128)
        x = res_block(x, 128)

        heads = []
        # Objectness: Use logit and and calculate sigmoid in loss for numerical
        # stability, initialize to favour no object as this otherwise dominates
        # training in the first epochs
        heads.append(
            tf.keras.layers.Conv2D(
                1,
                1,
                name="o",
                activation="linear",
                bias_initializer=tf.keras.initializers.Constant(-4.0),
            )(x)
        )
        # x,y in within grid cell
        heads.append(tf.keras.layers.Conv2D(1, 1, name="x", activation="sigmoid")(x))
        heads.append(tf.keras.layers.Conv2D(1, 1, name="y", activation="sigmoid")(x))
        # Width, Height in units of grid cells, cannot be larger than input image
        heads.append(
            tf.keras.layers.Rescaling(16, name="w")(
                tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(x)
            )
        )
        heads.append(
            tf.keras.layers.Rescaling(16, name="h")(
                tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(x)
            )
        )
        # Angle
        heads.append(tf.keras.layers.Conv2D(1, 1, name="a", activation="sigmoid")(x))
        # Class (=posture), sigmoid is calculated in loss
        heads.append(tf.keras.layers.Conv2D(1, 1, name="c", activation="linear")(x))

        head = tf.keras.layers.concatenate(heads, axis=-1)

        self.model = tf.keras.Model(inputs, head)

        if weights != None:
            if weights in pretrained_models.keys():
                self.model.load_weights(
                    resources.files("andap").joinpath("models", weights + ".weights.h5")
                )
                self.obj_thresh = pretrained_models[weights]["objectness_threshold"]
                self.overlap_thresh = pretrained_models[weights]["overlap_threshold"]
                self.class_thresh = pretrained_models[weights]["classification_threshold"]
            else:
                self.model.load_weights(weights)

    def train(
        self,
        dg_train,
        dg_val,
        save_path,
        epochs=200,
        object_loss_scale=45,
        kld_loss_scale=1.0,
        class_loss_scale=0.065,
        obj_alpha=0.25,
        obj_gamma=2.0,
        class_alpha=0.25,
        class_gamma=2.0,
        learning_rate=None,
        optimizer=None,
        rng_seed=0,
        **kwargs,
    ):
        """
        Train model with annotated data.

        Args:
            dg_train (data_generator.DataGenAugment): Data generator for training dataset.
            dg_val (data_generator.DataGen): Data generator for validation dataset.
            save_path (str): Path where weights will be saved.
            epochs (int, optional): Number of epochs to train. Defaults to 200.
            object_loss_scale (float, optional): Objectness loss is scaled by this factor. Defaults to 45.
            kld_loss_scale (float, optional): KLD loss is scaled by this factor. Defaults to 1.0.
            class_loss_scale (float, optional): Class loss is scaled by this factor. Defaults to 0.065.
            obj_alpha (float, optional): Alpha for objectness focal loss. Defaults to 0.25.
            obj_gamma (float, optional): Gamma for objectness focal loss. Defaults to 2.0.
            class_alpha (float, optional): Alpha for classification focal loss. Defaults to 0.25.
            class_gamma (float, optional): Gamma for classification focal loss. Defaults to 2.0.
            learning_rate (float or keras.optimizers.schedules.LearningRateSchedule, optional): If None (default), use CosineDecay.
            optimizer (tf.keras.optimizers instance): Optimizer to use for training. If None (default), use AdamW.
            rng_seed (int, optional): Sets the the tensorflow global random seed to ensure that training is reproducible.. Defaults to 0.

        Returns:
            keras.callbacks.History: Training history containing loss and metric values for all epochs.
        """

        tf.keras.utils.set_random_seed(rng_seed)

        loss = Loss(
            object_scale=object_loss_scale,
            kld_scale=kld_loss_scale,
            class_scale=class_loss_scale,
            batch_size=dg_train.batch_size,
            obj_alpha=obj_alpha,
            obj_gamma=obj_gamma,
            class_alpha=class_alpha,
            class_gamma=class_gamma,
        )

        if learning_rate is None:
            learning_rate = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=0.0,
                decay_steps=len(dg_train) * epochs,
                warmup_target=1e-3,
                warmup_steps=500,
                alpha=0.01,
            )

        if optimizer is None:
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=learning_rate,
                use_ema=True,
                clipnorm=1.0,
            )

        # Exclude bias and normalization layer params from weight decay
        no_decay_params = [
            w
            for w in self.model.trainable_weights
            if np.any([n in w.name for n in ("bias", "gamma", "beta")])
        ]
        optimizer.exclude_from_weight_decay(var_list=no_decay_params)

        self.model.compile(
            loss=loss,
            metrics=[loss.obj_loss, loss.kld_loss, loss.class_loss],
            optimizer=optimizer,
        )

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(save_path, "model.weights.h5"),
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode="min",
        )
        term = tf.keras.callbacks.TerminateOnNaN()

        callbacks = [checkpoint, term]

        return self.model.fit(
            dg_train,
            validation_data=dg_val,
            epochs=epochs,
            callbacks=callbacks,
            **kwargs,
        )

    def get_precision_recall_confusion(
        self,
        dg,
        objectness_thresholds,
        overlap_thresholds,
        class_thresholds,
        assignment_overlap_thresh=0.4,
    ):
        """
        Computes precision and recall for the object detection and confusion matrix for the posture classification on an annotated dataset.

        For precision and recall it only matters how many of the predicted ellipses match with a ground truth penguin, posture is not considered.
        For the posture classification confusion matrix, only true positives from object detection are considered.
        Grid cells at the boundary of the sample image are ignored because these are not used in the detection on the full images either.
        Args:
            dg (data_generator.DataGen or data_generator.DataGenAugment): Data generator for the annotated dataset to use.
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
        tp = np.zeros(
            (len(objectness_thresholds), len(overlap_thresholds)), dtype="int"
        )
        # Number of false positives
        fp = np.zeros(
            (len(objectness_thresholds), len(overlap_thresholds)), dtype="int"
        )
        # Number of false negatives
        fn = np.zeros(
            (len(objectness_thresholds), len(overlap_thresholds)), dtype="int"
        )

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

        # Get predictions and ground truth for each sample from data generator
        pred = []
        true = []
        for i in range(dg.__len__()):
            X, yt = dg.__getitem__(i)
            yp = self.model(X)

            # Discard grid cells at sample boundary and reshape
            yp = np.reshape(transform_output(yp)[:, 1:-1, 1:-1, :], (-1, 14 * 14, 7))
            yt = np.reshape(transform_output(yt)[:, 1:-1, 1:-1, :7], (-1, 14 * 14, 7))

            for imIdx in range(len(X)):
                pred.append(yp[imIdx])
                true.append(yt[imIdx])

        pred = np.array(pred)
        true = np.array(true)

        # In ground truth only grid cells with objectness 1 contain an object
        true_mask = true[..., 0] > 0.5

        xi, yi = np.meshgrid(np.arange(0, 14 * 14), np.arange(0, 14 * 14))

        ## Non-max suppression for prediction
        kld_ = kld(pred[:, xi], pred[:, yi])
        overlap = 1 / (1 + kld_)
        objectness_pred = pred[..., 0]
        # Get for each ellipse maximum overlap with another ellipse that has higher objectness
        mask = objectness_pred[:, xi] > objectness_pred[:, yi]
        max_overlap = np.max(np.where(mask, overlap, 0), axis=-1)

        for i, j in tqdm(
            itertools.product(
                range(len(objectness_thresholds)), range(len(overlap_thresholds))
            ),
            total=len(objectness_thresholds) * len(overlap_thresholds),
        ):
            obj_thresh = objectness_thresholds[i]
            overlap_thresh = overlap_thresholds[j]

            # Only take predicted objects with high enough objectness that are not
            # overlapped by higher objectness ellipses
            pred_mask = (max_overlap < overlap_thresh) & (objectness_pred > obj_thresh)

            for s in range(len(pred)):
                if not np.any(true_mask[s] | pred_mask[s]):
                    continue

                # Predicted and ground truth ellipses for sample s
                ps = pred[s][pred_mask[s]]
                ts = true[s][true_mask[s]]

                # Assign predicted to ground truth ellipses to maximise overlap
                # between matched pairs
                xx, yy = np.meshgrid(np.arange(len(ps)), np.arange(len(ts)))
                overlap_true_pred = 1 / (1 + kld(ts[yy], ps[xx]))
                true_idx, pred_idx = scipy.optimize.linear_sum_assignment(
                    overlap_true_pred, maximize=True
                )
                matched_true_idx = true_idx[
                    overlap_true_pred[true_idx, pred_idx] > assignment_overlap_thresh
                ]
                matched_pred_idx = pred_idx[
                    overlap_true_pred[true_idx, pred_idx] > assignment_overlap_thresh
                ]

                tp[i, j] += len(matched_true_idx)
                fp[i, j] += len(ps) - len(matched_true_idx)
                fn[i, j] += len(ts) - len(matched_true_idx)

                for k in range(len(class_thresholds)):
                    true_class = ts[matched_true_idx, 6] > 0.5
                    pred_class = ps[matched_pred_idx, 6] > class_thresholds[k]

                    confusion_matrix[i, j, k, 0, 0] += np.sum(
                        (~true_class & ~pred_class)
                    )
                    confusion_matrix[i, j, k, 0, 1] += np.sum(
                        (~true_class & pred_class)
                    )
                    confusion_matrix[i, j, k, 1, 0] += np.sum(
                        (true_class & ~pred_class)
                    )
                    confusion_matrix[i, j, k, 1, 1] += np.sum((true_class & pred_class))

        prec = tp / (tp + fp)
        rec = tp / (tp + fn)

        return prec, rec, tp, fp, fn, confusion_matrix

    def f1_acc_thresholding(
        self,
        dg,
        objectness_thresholds=None,
        overlap_thresholds=None,
        class_thresholds=None,
        assignment_overlap_thresh=0.4,
    ):
        """
        Optimizes F1 score on given dataset by choosing objectness and overlap thresholds for non-max suppression, and accuracy of posture classification by choosing the decision threshold.

        F1 score only considers object detection and not posture classification.
        For the classification accuracy, only true positives from object detection are considered.
        Grid cells at the boundary of the sample image are ignored because these are not used in the detection on the full images either.
        A grid search is used to determine the optimal values for the three threshold attributes of the Model object, which are set in the end.
        Args:
            dg (data_generator.DataGen or data_generator.DataGenAugment): Data generator for dataset to use.
            objectness_thresholds (ndarray, optional): 1D-array of objectness thresholds to be tested. If None (default), this will be set to np.linspace(0,1,51).
            overlap_thresholds (ndarray, optional): 1D-array of overlap thresholds for non-max suppression to be tested. If None (default), this will be set to np.linspace(0,1,51).
            class_thresholds (ndarray, optional): 1D-array of classification thresholds to be tested. If None (default), this will be set to np.linspace(0,1,51).
            assignment_overlap_thresh (float): If a predicted ellipse overlaps with a ground truth object by more than this value, it will be considered a true positive in the calculation of the F1 score. Defaults to 0.4.

        Returns:
            f1 (float): Object detection F1 score for best objectness and overlap threshold.
            accuracy (float): Posture classification accuracy for best classification threshold (and best objectness & overlap thresholds).
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

        prec, rec, tp, fp, fn, class_confusion = self.get_precision_recall_confusion(
            dg,
            objectness_thresholds,
            overlap_thresholds,
            class_thresholds,
            assignment_overlap_thresh,
        )
        f1 = 2 / (1 / rec + 1 / prec)

        obj_thresh_idx, overlap_thresh_idx = np.unravel_index(
            np.nanargmax(f1), f1.shape
        )
        self.obj_thresh = objectness_thresholds[obj_thresh_idx]
        self.overlap_thresh = overlap_thresholds[overlap_thresh_idx]
        print(
            f"Best F1-score {f1[obj_thresh_idx, overlap_thresh_idx]:.4f} "
            f"({tp[obj_thresh_idx, overlap_thresh_idx]} true positives, "
            f"{fp[obj_thresh_idx, overlap_thresh_idx]} false positives and "
            f"{fn[obj_thresh_idx, overlap_thresh_idx]} false negatives) "
            f"reached with obj_thresh={objectness_thresholds[obj_thresh_idx]} and "
            f"overlap_thresh={overlap_thresholds[overlap_thresh_idx]}."
        )

        cf = class_confusion[obj_thresh_idx, overlap_thresh_idx]
        accuracy = (cf[:, 0, 0] + cf[:, 1, 1]) / np.sum(cf, axis=(1, 2))
        class_thresh_idx = np.unravel_index(np.nanargmax(accuracy), accuracy.shape)
        self.class_thresh = class_thresholds[class_thresh_idx]
        print(
            f"Best classification accuracy {accuracy[class_thresh_idx]:.4f} "
            f"reached with class_thresh={class_thresholds[class_thresh_idx]}."
        )

        return f1[obj_thresh_idx, overlap_thresh_idx], accuracy[class_thresh_idx]

    def get_prediction(self, X, discard_sides=0, non_max_suppression=True):
        """
        Get prediction from model, transform outputs and run non-max suppression step.

        Args:
            X (ndarray): Batch of input images.
            discard_sides (int, optional): Number of grid cells to discard from each side. Defaults to 0.
            non_max_suppression (bool, optional): If True (default), performs non-max suppression step.

        Returns:
            ndarray: Array of shape (batch_size, 256, 7) with predictions for all grid cells when discard_sides is 0. In general, the grid dimension is (16 - 2*discard_sides)**2.
        """

        pred = transform_output(self.model(X)).numpy()

        if discard_sides > 0:
            pred = pred[:, discard_sides:-discard_sides, discard_sides:-discard_sides]

        pred = np.reshape(pred, (pred.shape[0], -1, 7))

        if not non_max_suppression:
            return pred

        assert self.obj_thresh != None, (
            "Need to set objectness threshold for non-max suppression."
        )

        assert self.overlap_thresh != None, (
            "Need to set overlap threshold for non-max suppression."
        )

        xx, yy = np.meshgrid(np.arange(pred.shape[1]), np.arange(pred.shape[1]))
        for imIdx in range(len(X)):
            kld_ = kld(pred[imIdx, xx], pred[imIdx, yy])
            overlap_matrix = 1 / (1 + kld_)
            objectness = pred[imIdx, :, 0]
            # Get for each ellipse maximum overlap with another ellipse that has higher objectness
            mask = objectness[xx] > objectness[yy]
            max_overlap = np.max(np.where(mask, overlap_matrix, 0), axis=1)
            pred[imIdx, :, 0] = np.where(
                (max_overlap < self.overlap_thresh) & (objectness > self.obj_thresh),
                pred[imIdx, :, 0],
                0,
            )

        return pred
