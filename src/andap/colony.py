import os
import numpy as np
import scipy
from tqdm import tqdm
import PIL.Image
import pandas
import geopandas
import shapely
import warnings
import rasterio
import pyproj.crs

from . import model
from . import utils
from . import pair_detection


# Disable image size limitation to load large orthomosaics
PIL.Image.MAX_IMAGE_PIXELS = None


class Colony:
    def __init__(
        self,
        name,
        results_dir,
        image_path=None,
        crs=None,
        penguins_fname=None,
        shapes_fname=None,
        colony_penguins_fname=None,
        nests_fname=None,
    ):
        """
        Creates colony object that is used as a basis for all steps of analysis.

        Args:
            name (str): Name of the colony. Used in filenames.
            results_dir (str): Path to directory where results will be stored.
            image_path (str, optional): Path to GeoTIFF image of the colony. Defaults to None, but either image_path or crs must be set.
            crs (obj, optional): Coordinate Reference System in which all computations will be performed, specified in a format accepted by pyproj.crs.from_user_input(). Defaults to None and will be overriden if image_path is set.
            penguins_fname (str, optional): Filename for storing detected penguins. If None (default), the filename will be of the form name_penguins.gpkg.
            shapes_fname (str, optional): Filename for storing subcolony shapes. If None (default), the filename will be of the form name_shapes.gpkg.
            colony_penguins_fname (str, optional): Filename for storing penguins inside colony boundaries. If None (default), the filename will be of the form name_colony_penguins.gpkg.
            nests_fname (str, optional): Filename for storing nest positions. If None (default), the filename will be of the form name_nests.gpkg.
        """
        self.name = name
        self.image_path = image_path
        self.results_dir = results_dir

        os.makedirs(self.results_dir, exist_ok=True)

        if penguins_fname is None:
            self.penguins_fname = self.name + "_penguins.gpkg"
        else:
            self.penguins_fname = penguins_fname

        if shapes_fname is None:
            self.shapes_fname = self.name + "_shapes.gpkg"
        else:
            self.shapes_fname = shapes_fname

        if colony_penguins_fname is None:
            self.colony_penguins_fname = self.name + "_colony_penguins.gpkg"
        else:
            self.colony_penguins_fname = colony_penguins_fname

        if nests_fname is None:
            self.nests_fname = self.name + "_nests.gpkg"
        else:
            self.nests_fname = nests_fname

        assert crs != None or image_path != None, "Need either image_path or crs."

        if image_path != None:
            with rasterio.open(self.image_path) as img:
                self.crs = img.crs
                self.img_transform = img.transform

            if abs(self.img_transform.a) != abs(self.img_transform.e):
                warnings.warn(
                    f"X and Y pixel size of GeoTIFF differ by factor of {self.img_transform.a / self.img_transform.e}. This will not be properly accounted for in run_detection()."
                )
        else:
            self.crs = pyproj.crs.from_user_input(crs)

        assert self.crs.linear_units_factor[1] == 1.0, (
            "CRS must be projected and using units of meters."
        )

    def run_detection(
        self, od_model, batch_size=128, restrict_regions=None, save_model_output=False
    ):
        """
        Runs penguin detection on image and writes result to the penguins file.

        Args:
            od_model (andap.model.Model): The object detection model to use for the detection.
            batch_size (int, optional): Number of image tiles fed into the network simultaneously. Defaults to 128.
            save_model_output (bool, optional): Save penguins with objectness and class output values from the model in another file. This is only useful when evaluating model performance on test data with different thresholds. Defaults to False.
            restrict_regions (shapely.Geometry or str, optional): Only detect penguins within this region. Can be either path to GeoPackage file or shapely geometry in world coordinates. If None (default), the full image is analysed.
        """
        # Size of grid cells in pixels
        gc_size = 4

        # Width/height of neural network's input/output in number of grid cells
        n_gc = 16

        # Number of bounding boxes to discard at each side of NN's output
        discard_sides = 1

        img_data = np.array(PIL.Image.open(self.image_path).convert("RGBA"))

        # Make transparent (nodata) pixels blue, as the model has no alpha input
        # and might confuse black background for penguins at the boundary
        img_data[img_data[..., 3] < 255, :3] = (0, 0, 255)

        # Dimensions of image in units of grid cells
        xg = int(np.ceil(img_data.shape[1] / gc_size))
        yg = int(np.ceil(img_data.shape[0] / gc_size))

        ellipses = []

        xgrid, ygrid = np.meshgrid(
            np.arange(0, xg, n_gc - 2 * discard_sides),
            np.arange(0, yg, n_gc - 2 * discard_sides),
        )
        xgrid = xgrid.flatten()
        ygrid = ygrid.flatten()

        tile_coords = (
            np.stack(
                [
                    np.stack([xgrid, ygrid], axis=1),
                    np.stack([xgrid + n_gc, ygrid], axis=1),
                    np.stack([xgrid + n_gc, ygrid + n_gc], axis=1),
                    np.stack([xgrid, ygrid + n_gc], axis=1),
                ],
                axis=1,
            )
            * gc_size
        )

        # Check which tiles are partially within restrict_regions
        if restrict_regions is not None:
            if isinstance(restrict_regions, str):
                rr_gdf = geopandas.read_file(restrict_regions).to_crs(self.crs)
                restrict_regions = shapely.unary_union(rr_gdf.geometry)
            restrict_regions = shapely.affinity.affine_transform(
                restrict_regions, (~self.img_transform).to_shapely()
            )

            # Use slightly larger regions to improve detection on boundaries,
            # penguins outside the orignal regions are removed in the end
            restrict_regions_buf = shapely.buffer(
                restrict_regions, 5, cap_style="square"
            )
            shapely.prepare(restrict_regions_buf)

            inside_tiles_indices = []
            for i in range(len(xgrid)):
                tile_rect = shapely.Polygon(tile_coords[i])
                if tile_rect.intersects(restrict_regions_buf):
                    inside_tiles_indices.append(i)

            xgrid = xgrid[inside_tiles_indices]
            ygrid = ygrid[inside_tiles_indices]
            tile_coords = tile_coords[inside_tiles_indices]

        tiles = np.zeros(
            (len(xgrid), n_gc * gc_size, n_gc * gc_size, 4), dtype=img_data.dtype
        )
        # Set background to blue for tiles not covered fully by image
        tiles[..., 2] = 255
        for i in range(len(xgrid)):
            tile = img_data[
                ygrid[i] * gc_size : (ygrid[i] + n_gc) * gc_size,
                xgrid[i] * gc_size : (xgrid[i] + n_gc) * gc_size,
            ]
            h, w = tile.shape[:2]
            tiles[i, :h, :w] = tile

        # Remove tiles that are empty
        nonempty_mask = np.any(tiles[..., 3] == 255, axis=(1, 2))

        tiles = tiles[nonempty_mask]
        xgrid = xgrid[nonempty_mask]
        ygrid = ygrid[nonempty_mask]
        tile_coords = tile_coords[nonempty_mask]

        # Find grid cells containing only opaque pixels, predictions from all others
        # will later be discarded
        nonempty_gc_mask = np.all(
            tiles[..., 3].reshape(-1, 16, 4, 16, 4) == 255, axis=(2, 4)
        )

        tiles = tiles[..., :3]

        for tile_idx in tqdm(range(0, len(tiles), batch_size)):
            batch_indices = range(tile_idx, min(tile_idx + batch_size, len(tiles)))
            input_imgs = tiles[batch_indices]
            batch_tile_coords = tile_coords[batch_indices]

            if save_model_output:
                pred = od_model.get_prediction(
                    input_imgs, discard_sides=discard_sides, non_max_suppression=False
                )
            else:
                pred = od_model.get_prediction(input_imgs, discard_sides=discard_sides)

            pred[..., 0] *= nonempty_gc_mask[
                batch_indices,
                discard_sides:-discard_sides,
                discard_sides:-discard_sides,
            ].reshape(len(batch_indices), (16 - 2 * discard_sides) ** 2)

            for i in range(len(pred)):
                pred[i, :, 1] += batch_tile_coords[i, 0, 0]
                pred[i, :, 2] += batch_tile_coords[i, 0, 1]
            obj_mask = pred[:, :, 0] > 0
            if np.any(obj_mask) or save_model_output:
                pred = pred[obj_mask | save_model_output]
                ellipses.extend(pred)

        assert len(ellipses) > 0, "No penguins detected."
        ellipses = np.array(ellipses)

        # Remove predictions outside restrict_regions
        if restrict_regions is not None:
            shapely.prepare(restrict_regions)
            contained_mask = shapely.intersects_xy(restrict_regions, ellipses[:, 1:3])
            ellipses = ellipses[contained_mask]

        print(
            f"{np.sum(ellipses[:, 6] > od_model.class_thresh)} lying and "
            f"{np.sum(ellipses[:, 6] <= od_model.class_thresh)} standing "
            "penguins detected."
        )

        # When raw model output is wanted, we are done here as no non-max suppression is necessary
        if save_model_output:
            coords = utils.apply_affine(
                self.img_transform, np.asarray(ellipses[:, 1:3], dtype="float64")
            )
            ellipses[:, 3:5] *= abs(self.img_transform.a)
            ellipses[:, 5] *= 180 / np.pi

            df = pandas.DataFrame(
                ellipses[:, [0, 3, 4, 5, 6]],
                columns=[
                    "objectness",
                    "width [m]",
                    "height [m]",
                    "angle [deg]",
                    "posture",
                ],
            )
            utils.save_points(
                os.path.join(self.results_dir, "model_output_" + self.penguins_fname),
                self.crs,
                coords,
                df,
            )
            return

        print(
            "Starting non-max suppression step to remove overlapping ellipses at tile boundaries."
        )

        # Only check overlap of those ellipses that are close together - others are very
        # unlikely to get filtered by non-max suppression
        tree = scipy.spatial.cKDTree(ellipses[:, 1:3])
        pairs = tree.query_pairs(gc_size * 2)

        delete_mask = np.zeros(len(ellipses), dtype=bool)

        for i, pi in enumerate(pairs):
            if delete_mask[pi[0]] or delete_mask[pi[1]]:
                continue

            e1 = ellipses[pi[0]]
            e2 = ellipses[pi[1]]
            overlap = 1 / (1 + model.kld(e1, e2))
            if overlap > od_model.overlap_thresh:
                if e1[0] > e2[0]:
                    delete_mask[pi[1]] = True
                else:
                    delete_mask[pi[0]] = True

        print(f"{np.sum(delete_mask)} penguins removed in global non-max suppression.")
        ellipses = ellipses[~delete_mask]
        coords = utils.apply_affine(
            self.img_transform, np.asarray(ellipses[:, 1:3], dtype="float64")
        )
        ellipses[:, 3:5] *= abs(self.img_transform.a)
        ellipses[:, 5] *= 180 / np.pi
        ellipses[:, 6] = np.where(ellipses[:, 6] > od_model.class_thresh, 1, 0)
        df = pandas.DataFrame(
            ellipses[:, 3:],
            columns=["width [m]", "height [m]", "angle [deg]", "posture"],
        )
        utils.save_points(
            os.path.join(self.results_dir, self.penguins_fname), self.crs, coords, df
        )

    def delineate_subcolonies(self, alpha_radius=0.75, buffer=None):
        """
        Finds polygons covering subcolony areas using alpha shapes.

        Args:
            alpha_radius (float, optional): Alpha radius in meters. Defaults to 0.75.
            buffer (float, optional): Distance in meters by which alpha shapes are buffered. If None (default), it will be set to alpha_radius/2.

        Returns:
            shapely.geometry.base.GeometrySequence: The resulting subcolony shapes.
        """
        if buffer is None:
            buffer = alpha_radius / 2

        penguins = utils.load_penguins(
            os.path.join(self.results_dir, self.penguins_fname), self.crs
        )

        polygons = utils.alpha_shape(penguins[:, :2], 1 / alpha_radius, buffer)

        gdf = geopandas.GeoDataFrame({"geometry": polygons}, crs=self.crs)
        gdf.to_file(os.path.join(self.results_dir, self.shapes_fname), driver="GPKG")

        return polygons

    def filter_outside_subcolonies(self):
        """
        Removes all penguins outside the subcolony delineation polygons.
        """
        penguins = utils.load_penguins(
            os.path.join(self.results_dir, self.penguins_fname), self.crs
        )

        gdf = geopandas.read_file(
            os.path.join(self.results_dir, self.shapes_fname)
        ).to_crs(self.crs)
        polygons = shapely.unary_union(gdf.geometry)
        shapely.prepare(polygons)
        inside_mask = shapely.intersects_xy(polygons, penguins[:, 0], penguins[:, 1])
        colony_penguins = penguins[inside_mask]
        utils.save_penguins(
            os.path.join(self.results_dir, self.colony_penguins_fname),
            self.crs,
            colony_penguins,
        )

    def run_pair_removal(self, threshold_dists=None, plot=False):
        """
        Identifies and removes non-nesting penguins based on their distance to neighbors.

        Args:
            threshold_dists (tuple[float,float,float] optional): Minimum allowed inter-nest distances for pairs of 1) two lying, 2) two standing, and 3) one lying and one standing penguin. If None (default), the thresholds are determined automatically to split the mixed distributions of the mutual nearest-neighbor distances.
            plot (bool, optional): If True, the automatically determined thresholds will be visualized together with the distributions of mutual nearest-neighbor distances. Defaults to False.
        """
        penguins = utils.load_penguins(
            os.path.join(self.results_dir, self.colony_penguins_fname), self.crs
        )

        # Center coordinates around the mean to avoid numerical issues
        penguins_c = np.copy(penguins)
        penguins_c[:, :2] -= np.mean(penguins_c[:, :2], axis=0)

        tree = scipy.spatial.cKDTree(penguins_c[:, :2])
        distances, neighbors = tree.query(penguins_c[:, :2], k=2)
        distances = distances[:, 1]
        neighbors = neighbors[:, 1]

        if threshold_dists is None:
            ll_dists, ss_dists, ls_dists = pair_detection.get_distances_by_type(
                penguins_c, distances, neighbors
            )
            threshold_dists = (
                pair_detection.find_threshold(ll_dists, plot=plot, title="Lying-Lying"),
                pair_detection.find_threshold(
                    ss_dists, plot=plot, title="Standing-Standing"
                ),
                pair_detection.find_threshold(
                    ls_dists, plot=plot, title="Lying-Standing"
                ),
            )
            print(
                f"Removing pairs closer than {threshold_dists[0]:.4f}m for LL, "
                f"{threshold_dists[1]:.4f}m for SS and "
                f"{threshold_dists[2]:.4f}m for LS."
            )

        nests = pair_detection.remove_pairs(
            threshold_dists, penguins_c, distances, neighbors
        )
        nests[:, :2] += np.mean(penguins[:, :2], axis=0)

        utils.save_penguins(
            os.path.join(self.results_dir, self.nests_fname),
            self.crs,
            nests,
        )
