import numpy as np
from numba import jit
from scipy.spatial import Delaunay
import shapely
import geopandas
import pandas
import rasterio


posture_map = {"standing": 0, "lying": 1}


def save_points(file_path, crs, points, data=None):
    """
    Save points with additional data to GeoPackage file.

    Args:
        file_path (str): Path to file where data is saved.
        crs (obj): Coordinate Reference System specified in a format accepted by pyproj.crs.from_user_input().
        points (ndarray): Points in world coordinates.
        data (pandas.DataFrame, optional): Additional data associated with points. Defaults to None.
    """
    points = np.asarray(points, dtype="float64")
    gdf = geopandas.GeoDataFrame(
        data, geometry=geopandas.points_from_xy(points[:, 0], points[:, 1]), crs=crs
    )
    gdf.to_file(file_path, driver="GPKG")


def load_points(file_path, crs):
    """
    Load points and associated GeoDataFrame from GeoPackage file.

    Args:
        file_path (str): Path to file from which data is loaded.
        crs (obj): Coordinate Reference System specified in a format accepted by pyproj.crs.from_user_input().

    Returns:
        ndarray: Points in world coordinates.
        geopandas.GeoDataFrame: Additional data associated with points.
    """
    gdf = geopandas.read_file(file_path).to_crs(crs)
    points = np.stack([gdf.geometry.x, gdf.geometry.y], axis=1)
    return points, gdf


def save_penguins(file_path, crs, penguins):
    """
    Save penguin data (positions + posture) to GeoPackage file.

    Args:
        file_path (str): Path to file where data is saved.
        crs (obj): Coordinate Reference System specified in a format accepted by pyproj.crs.from_user_input().
        penguins (ndarray): (N,3) array containing position in world coordinates and posture value for N penguins.
    """
    save_points(
        file_path,
        crs,
        penguins[:, :2],
        pandas.DataFrame(penguins[:, 2], columns=["posture"]),
    )


def load_penguins(file_path, crs):
    """
    Load penguin data (positions + posture) from GeoPackage file.

    Args:
        file_path (str): Path to file from which penguin data is loaded.
        crs (obj): Coordinate Reference System specified in a format accepted by pyproj.crs.from_user_input().

    Returns:
        penguins (ndarray): (N,3) array containing position in world coordinates and posture value for N penguins.
    """
    points, gdf = load_points(file_path, crs)
    return np.hstack((points, np.array([gdf["posture"]]).T))


def apply_affine(transform, coords):
    """
    Apply affine transformation to array of points.

    Args:
        transform (affine.Affine): rasterio/affine transformation matrix.
        coords (ndarray): Array of coordinates. Last axis must contain (x,y) pairs.

    Returns:
        ndarray: Transformed coordinates.
    """
    x = coords[..., 0]
    y = coords[..., 1]

    xt = transform.a * x + transform.b * y + transform.c
    yt = transform.d * x + transform.e * y + transform.f

    return np.stack([xt, yt], axis=-1)


def alpha_shape(points, alpha, buffer=0):
    """
    Computes alpha shape of points.

    Args:
        points (ndarray): (n,2) array of points.
        alpha (float): Alpha parameter, which is the inverse of the alpha radius.
        buffer (int, optional): Distance by which the alpha shape gets buffered. Defaults to 0.

    Returns:
        shapely.geometry.base.GeometrySequence: The resulting alpha shape
    """

    @jit(nopython=True)
    def alpha_shape_jit(vert, points, alpha):
        out = np.empty((vert.shape[0], 3, 2), dtype=points.dtype)
        count = 0

        for ia, ib, ic in vert:
            a = np.linalg.norm(points[ia] - points[ib])
            b = np.linalg.norm(points[ib] - points[ic])
            c = np.linalg.norm(points[ic] - points[ia])

            # Heron's formula for area of triangle
            s = (a + b + c) / 2.0
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))

            if area > 0:
                circumradius = a * b * c / (4.0 * area)

                if circumradius < 1 / alpha:
                    out[count, 0] = points[ia]
                    out[count, 1] = points[ib]
                    out[count, 2] = points[ic]
                    count += 1

        return out[:count]

    mean_coords = np.mean(points, axis=0)
    triangles = alpha_shape_jit(
        Delaunay(points - mean_coords).simplices, points - mean_coords, alpha
    )
    triangle_polygons = [shapely.geometry.Polygon(t + mean_coords) for t in triangles]
    union = shapely.ops.unary_union(triangle_polygons)
    buffered = union.buffer(buffer)
    if type(buffered) == shapely.geometry.multipolygon.MultiPolygon:
        return buffered.geoms
    else:
        return shapely.geometry.base.GeometrySequence(buffered)


def points_in_ellipse(x, y, xe, ye, w, h, a):
    """
    Check if points are inside an ellipse.

    Args:
        x (ndarray): x coordinates of points.
        y (ndarray): y coordinates of points.
        xe (float): x coordinate of ellipse.
        ye (float): y coordinate of ellipse.
        w (float): Width of ellipse.
        h (float): Heigth of ellipse.
        a (float): Angle of ellipse (radians)

    Returns:
        ndarray: Boolean array indicating which points (x,y) are inside the ellipse.
    """
    dx = x - xe
    dy = y - ye
    x_rot = dx * np.cos(-a) - dy * np.sin(-a)
    y_rot = dx * np.sin(-a) + dy * np.cos(-a)
    with np.errstate(divide="ignore"):
        return (x_rot**2 / (w / 2) ** 2 + y_rot**2 / (h / 2) ** 2) <= 1.0
