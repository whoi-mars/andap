import numpy as np
import scipy.stats
import scipy.signal
import matplotlib.pyplot as plt

from .utils import posture_map


def get_distances_by_type(penguins, dists, nn_indices, mutual_only=True):
    """
    Get (mutual) nearest neighbor distances for different types of neighbors (both penguins lying, both standing or standing-lying)

    Args:
        penguins (ndarray): (N,3) array containing penguin coordinates and posture.
        dists (ndarray): Nearest neighbor distances of penguins.
        nn_indices (ndarray): The index of the nearest neighbor of each penguin.
        mutual_only (bool): If True (default), only count two penguins as nearest neighbors if both have no other neighbor with smaller distance. If False, it suffices if one of them has the other as the closest neighbor.

    Returns:
        tuple of ndarray: Distances of LL-pairs, SS-pairs and LS-pairs.
    """
    points = penguins[:, :2]
    posture = penguins[:, 2]

    processed_mask = np.zeros(len(points), dtype=bool)

    ll_dists = []
    ss_dists = []
    ls_dists = []

    for i, dist in enumerate(dists):
        if processed_mask[i]:
            continue

        nn_idx = nn_indices[i]
        nns_nn_idx = nn_indices[nn_idx]

        if nns_nn_idx == i or not mutual_only:
            if (
                posture[i] == posture_map["standing"]
                and posture[nn_idx] == posture_map["lying"]
            ):
                ls_dists.append(dist)
            elif (
                posture[i] == posture_map["lying"]
                and posture[nn_idx] == posture_map["standing"]
            ):
                ls_dists.append(dist)

            elif (
                posture[i] == posture_map["lying"]
                and posture[nn_idx] == posture_map["lying"]
            ):
                ll_dists.append(dist)
            elif (
                posture[i] == posture_map["standing"]
                and posture[nn_idx] == posture_map["standing"]
            ):
                ss_dists.append(dist)

            processed_mask[i] = True

            if mutual_only:
                processed_mask[nn_idx] = True

    return ll_dists, ss_dists, ls_dists


def mixed_t_pdf(x, weight, df1, loc1, scale1, df2, loc2, scale2):
    """
    Probability density function for mixture of two Student's t distributions.

    Args:
        x (ndarray): Argument for the PDF.
        weight (float): Weight of first component in [0,1].
        df1 (float): Degrees of freedom parameter for first component.
        loc1 (float): Location parameter for first component.
        scale1 (float): Scale parameter for first component.
        df2 (float): Degrees of freedom parameter for second component.
        loc2 (float): Location parameter for second component.
        scale2 (float): Scale parameter for second component.

    Returns:
        ndarray: PDF evaluated at x.
    """
    pdf1 = scipy.stats.t.pdf(x, df1, loc=loc1, scale=scale1)
    pdf2 = scipy.stats.t.pdf(x, df2, loc=loc2, scale=scale2)
    return weight * pdf1 + (1 - weight) * pdf2


def mixed_t_nll(x, weight, df1, loc1, scale1, df2, loc2, scale2):
    """Negative log-likelihood of the mixture of two Student's t distributions.

    Args:
        x (ndarray): Observed data.
        weight (float): Weight of first component in [0,1].
        df1 (float): Degrees of freedom parameter for first component.
        loc1 (float): Location parameter for first component.
        scale1 (float): Scale parameter for first component.
        df2 (float): Degrees of freedom parameter for second component.
        loc2 (float): Location parameter for second component.
        scale2 (float): Scale parameter for second component.

    Returns:
        float: Mean negative log-likelihood for observed data.
    """
    y = mixed_t_pdf(x, weight, df1, loc1, scale1, df2, loc2, scale2)
    return -np.mean(np.log(y))


def fit_mixed_log_t(data, **fit_kwargs):
    """
    Maximum likelihood estimation of parameters for Student's t mixture model on log-transformed data.

    Args:
        data (ndarray): Observed data.
        **fit_kwargs (optional): Keyword arguments passed to scipy.optimize.minimize. If none are given, x0, bounds, method and options will be set to suitable default values.

    Returns:
        ndarray: The estimated parameters, in the same order as in mixed_t_pdf's arguments.
    """
    default_kwargs = {
        "x0": [
            0.5,            # Weight
            2,              # df1
            np.log10(0.4),  # loc1
            0.03,           # scale1
            2,              # df2
            np.log10(0.65), # loc2
            0.03            # scale2
        ],
        "bounds": [
            (0.05, 0.95),   # Weight
            (0, None),      # df1
            (None, None),   # loc1
            (None, None),   # scale1
            (0, None),      # df2
            (None, None),   # loc2
            (None, 0.1),    # scale2
        ],
        "method": "SLSQP",
        "options": {"maxiter": 5000},
    }
    merged_kwargs = {**default_kwargs, **fit_kwargs}

    log_data = np.log10(data)
    wrapped_nll = lambda params: mixed_t_nll(log_data, *params)
    result = scipy.optimize.minimize(wrapped_nll, **merged_kwargs)

    if not result["success"]:
        print("Mixed log t fit failed:", result["message"])

    return result["x"]


def find_threshold(dists, plot=False, title="", **fit_kwargs):
    """
    Finds distance threshold to distinguish neighboring nests from pairs of nesting and non-nesting penguins.

    Args:
        dists (ndarray): Nearest neighbor distances of penguins.
        plot (boolean, optional): If True, results are plotted (useful for debugging). Defaults to False.
        title (str, optional): Figure axes title if plot=True.
        **fit_kwargs (optional): Keyword arguments passed to scipy.optimize.minimize. If none are given, x0, bounds, method and options will be set to suitable default values.

    Returns:
        float: Distance threshold.
    """
    weight, df1, loc1, scale1, df2, loc2, scale2 = fit_mixed_log_t(dists, **fit_kwargs)

    cdf1 = lambda x: scipy.stats.t.cdf(np.log10(x), df1, loc=loc1, scale=scale1)
    cdf2 = lambda x: scipy.stats.t.cdf(np.log10(x), df2, loc=loc2, scale=scale2)
    if loc1 < loc2:
        class_error = lambda x: weight * (1 - cdf1(x)) - (1 - weight) * cdf2(x)
    else:
        class_error = lambda x: weight * (cdf1(x)) - (1 - weight) * (1 - cdf2(x))

    result = scipy.optimize.root(class_error, x0=(10**loc1 + 10**loc2) / 2)
    threshold = result["x"][0]

    if plot:
        fig, ax = plt.subplots()
        ax.set_title(title)

        bins = np.linspace(0, 2, 200)
        binsx = (bins[1:] + bins[:-1]) / 2
        hist, *_ = np.histogram(dists, bins=bins, density=True)
        ax.bar(
            binsx[binsx < threshold],
            hist[binsx < threshold],
            color="red",
            alpha=0.3,
            width=np.diff(bins)[binsx < threshold],
        )
        ax.bar(
            binsx[binsx > threshold],
            hist[binsx > threshold],
            color="blue",
            alpha=0.3,
            width=np.diff(bins)[binsx > threshold],
        )

        ax.vlines(
            [threshold],
            ymin=0,
            ymax=np.max(hist),
            color="black",
            linestyle="--",
            label="Threshold",
        )

        ls = np.linspace(0, 1.4, 1000)
        pdf1 = (
            weight
            * scipy.stats.t.pdf(np.log10(ls), df1, loc=loc1, scale=scale1)
            / np.abs(ls * np.log(10))
        )
        pdf2 = (
            (1 - weight)
            * scipy.stats.t.pdf(np.log10(ls), df2, loc=loc2, scale=scale2)
            / np.abs(ls * np.log(10))
        )

        if loc1 < loc2:
            ax.plot(ls, pdf1, label="Nest + non-nesting", color="Red")
            ax.plot(ls, pdf2, label="Neighboring nests", color="Blue")
        else:
            ax.plot(ls, pdf2, label="Nest + non-nesting", color="Red")
            ax.plot(ls, pdf1, label="Neighboring nests", color="Blue")

        ax.plot(ls, pdf1 + pdf2, label="Mixture Model", color="Black")

        ax.legend()
        ax.set_xlabel("Nearest Neighbor Distance [m]")
        ax.set_ylabel("Probability Density")
        ax.set_xlim(0, 1.4)
        plt.show()

    return threshold


def remove_pairs(threshold_dists, penguins, dists, nn_indices):
    """
    Replaces pairs of mutual nearest neighbors standing closer together than the supplied thresholds by a single marker.

    Args:
        threshold_dists (ndarray): List of distance thresholds for lying-lying, standing-standing and standing-lying pairs in that order.
        penguins (ndarray): (N,3) array containing penguin coordinates and posture.
        dists (ndarray): Nearest neighbor distances of penguins.
        nn_indices (ndarray): The index of the nearest neighbor of each penguin.

    Returns:
        ndarray: (M,3) array containing remaining penguins after pair removal. Posture information is carried over from original penguins, i.e. the entry for a united lying-lying pair will still be labeled as lying.
    """
    points = penguins[:, :2]
    posture = penguins[:, 2]

    processed_mask = np.zeros(len(points), dtype=bool)
    delete_mask = np.zeros(len(points), dtype=bool)

    new_points = []

    ll_counter = 0
    ss_counter = 0
    ls_counter = 0

    pair_indices = np.where(dists < np.max(threshold_dists))[0]

    for pi in pair_indices:
        if processed_mask[pi]:
            continue

        # Only treat pairs of mutual NNs
        nn_idx = nn_indices[pi]
        if nn_indices[nn_idx] == pi:
            # One partner standing, the other one lying -> take lying one as nest
            if (
                posture[pi] == posture_map["standing"]
                and posture[nn_idx] == posture_map["lying"]
                and dists[pi] < threshold_dists[2]
            ):
                delete_mask[pi] = True
                ls_counter += 1
            elif (
                posture[pi] == posture_map["lying"]
                and posture[nn_idx] == posture_map["standing"]
                and dists[pi] < threshold_dists[2]
            ):
                delete_mask[nn_idx] = True
                ls_counter += 1
            # Both lying/standing -> use midpoint
            elif (
                posture[pi] == posture_map["lying"]
                and posture[nn_idx] == posture_map["lying"]
                and dists[pi] < threshold_dists[0]
            ):
                new_points.append(
                    np.append((points[pi] + points[nn_idx]) / 2, posture_map["lying"])
                )
                delete_mask[pi] = True
                delete_mask[nn_idx] = True
                ll_counter += 1
            elif (
                posture[pi] == posture_map["standing"]
                and posture[nn_idx] == posture_map["standing"]
                and dists[pi] < threshold_dists[1]
            ):
                new_points.append(
                    np.append(
                        (points[pi] + points[nn_idx]) / 2, posture_map["standing"]
                    )
                )
                delete_mask[pi] = True
                delete_mask[nn_idx] = True
                ss_counter += 1

            processed_mask[pi] = True
            processed_mask[nn_idx] = True

    print(
        f"Found {ll_counter} lying-lying, {ss_counter} standing-standing and {ls_counter} standing-lying pairs. {np.sum(~delete_mask)} penguins are not part of a pair."
    )

    if len(new_points) > 0:
        return np.concatenate((penguins[~delete_mask], new_points))
    else:
        return penguins[~delete_mask]
