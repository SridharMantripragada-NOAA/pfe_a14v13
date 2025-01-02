import logging
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr

from .stats_tests import get_chi2test_2samp_all, get_kstest_2samp

logger = logging.getLogger(__name__)


def dist_haversine(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """
    Calculate the great-circle distance between two points on the Earth using the
    haversine formula.

    Args:
        lat1 (np.ndarray): Latitude(s) of the first point(s) in degrees.
        lon1 (np.ndarray): Longitude(s) of the first point(s) in degrees.
        lat2 (np.ndarray): Latitude(s) of the second point(s) in degrees.
        lon2 (np.ndarray): Longitude(s) of the second point(s) in degrees.

    Returns:
        np.ndarray: Distance(s) between the points in kilometers.
    """
    # Radius of the Earth in kilometers
    earth_radius_km = 6371.0

    # Convert latitudes and longitudes from degrees to radians
    lat1_rad, lon1_rad = np.deg2rad(lat1), np.deg2rad(lon1)
    lat2_rad, lon2_rad = np.deg2rad(lat2), np.deg2rad(lon2)

    # Compute differences in latitudes and longitudes
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    # Compute haversine formula
    a = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return earth_radius_km * c


def usekernel(h: np.ndarray, hmin: float, hmax: float) -> np.ndarray:
    """
    Apply a modified triweight kernel function to calculate weights based on the input
    range.

    Args:
        h (np.ndarray): Input array for which weights are calculated.
        hmin (float): Minimum value of the range.
        hmax (float): Maximum value of the range.

    Returns:
        np.ndarray: Weight array with values transitioning from 1 to 0 within
        [hmin, hmax].
    """
    if hmin < hmax:
        # Calculate weights using the modified triweight kernel function
        w = (1.0 - ((h - hmin) / (hmax - hmin)) ** 2) ** 3
    else:
        # Initialize weights as 1 if hmin equals hmax
        w = np.ones_like(h)

    # Apply conditions to enforce weights outside the range
    w[np.abs(h) > hmax] = 0.0  # Set weights to 0 outside the range
    w[np.abs(h) < hmin] = 1.0  # Set weights to 1 below hmin

    return w


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} execution time: {end_time - start_time:.6f} seconds")
        return result

    return wrapper


# @timing_decorator
def compute_distances(src_lat: float, src_lon: float, meta: pd.DataFrame) -> np.ndarray:
    """Compute haversine distances for all stations."""
    try:
        # Check if required columns exist in the DataFrame
        for column in ["LAT", "LON"]:
            if column not in meta.columns:
                logging.error(
                    "Column '%s' is missing in the provided meta DataFrame.", column
                )
                raise ValueError(
                    f"Column '{column}' is required in the meta DataFrame but"
                    "is missing."
                )

        return dist_haversine(src_lat, src_lon, meta["LAT"].values, meta["LON"].values)
    except Exception as e:
        logging.error("Error in computing distances: %s", e)
        raise


# @timing_decorator
def filter_neighbor_stations(meta: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Filter neighbors within the maximum search radius."""
    try:
        max_radius = config["max_radius"]

        # Check if the 'd' column exists in the DataFrame
        if "d" not in meta.columns:
            logging.error("Column 'd' is missing in the provided meta DataFrame.")
            raise ValueError(
                "Column 'd' is required in the meta DataFrame but is missing."
            )

        return meta[meta["d"] < max_radius].sort_values(by="d").copy()
    except KeyError as e:
        logging.error("Missing configuration key: %s", e)
        raise


# @timing_decorator
def compute_distance_weights(distances: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """Compute weights based on distance."""
    try:
        # Check if required configuration keys exist
        for key in ["min_radius", "max_radius"]:
            if key not in config:
                logging.error("Missing configuration key: '%s'", key)
                raise KeyError(f"Configuration key '{key}' is required but is missing.")

        return usekernel(distances, config["min_radius"], config["max_radius"])
    except Exception as e:
        logging.error("Error in computing distance weights: %s", e)
        raise


# @timing_decorator
def compute_elevation_range_and_obstacle_height(
    config: Dict[str, Any], sub_meta: pd.DataFrame, elev_srtm: xr.Dataset
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute elevation statistics between a central point and an array of
    neighbor points.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
        sub_meta (pd.DataFrame): DataFrame containing neighbor metadata.
        elev_srtm (xr.Dataset, optional): Elevation dataset.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Maximum-minimum elevation differences along paths (elev_stn_srtm_range).
            - Obstacle heights along paths (elev_stn_srtm_oh).
    """
    try:
        # Validate required columns in sub_meta
        for column in ["LAT", "LON"]:
            if column not in sub_meta.columns:
                logging.error(
                    "Missing required column '%s' in sub_meta DataFrame.", column
                )
                raise ValueError(
                    f"Column '{column}' is required but missing in sub_meta."
                )

        # Validate required keys in config
        if "obstacle_height_step" not in config:
            logging.error("Missing required configuration key: 'obstacle_height_step'")
            raise KeyError(
                "Configuration key 'obstacle_height_step' is required but missing."
            )

        # Extract key inputs
        latitude0, longitude0 = sub_meta.LAT.values[0], sub_meta.LON.values[0]
        latitudes, longitudes = sub_meta.LAT.values, sub_meta.LON.values
        fstep = config["obstacle_height_step"]

        # Extract longitude step size
        lon_values = elev_srtm["lon"].values
        dlon = abs(lon_values[0] - lon_values[-1]) / (len(lon_values) - 1)

        # Initialize results
        elev_stn_srtm_range: List[float] = []
        elev_stn_srtm_oh: List[float] = []

        # Process each neighbor point
        for lat1, lon1 in zip(latitudes, longitudes):
            # Compute path length and number of evaluation points
            distance = np.sqrt((latitude0 - lat1) ** 2 + (longitude0 - lon1) ** 2)
            num_points = max(round(distance / (fstep * dlon)), 1)

            # Generate path points
            lat_path = np.linspace(latitude0, lat1, num_points + 1)
            lon_path = np.linspace(longitude0, lon1, num_points + 1)

            # Extract elevations along the path
            elev_stn_srtm = [
                elev_srtm.sel(lat=lat, lon=lon, method="nearest").values.item()
                for lat, lon in zip(lat_path, lon_path)
            ]

            # Compute elevation statistics
            max_elev = max(elev_stn_srtm)
            min_elev = min(elev_stn_srtm)
            elev_stn_srtm_range.append(max_elev - min_elev)  # Max-min range
            elev_stn_srtm_oh.append(
                max_elev - max(elev_stn_srtm[0], elev_stn_srtm[-1])
            )  # Obstacle height

        return np.array(elev_stn_srtm_range), np.array(elev_stn_srtm_oh)

    except Exception as e:
        logging.error("Error in get_elevation_between_points: %s", e)
        raise


# @timing_decorator
def compute_mam_weights(
    sub_meta: pd.DataFrame, config: Dict[str, Any], mam0: float
) -> pd.DataFrame:
    """
    Compute MAM differences and weights for neighbors.

    Args:
        sub_meta (pd.DataFrame): DataFrame containing neighbor metadata.
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        pd.DataFrame: Updated metadata with computed MAM differences and weights.
    """
    try:
        sub_meta["dMAM"] = (
            200 * np.abs(sub_meta["usedMAM"] - mam0) / (sub_meta["usedMAM"] + mam0)
        )
        sub_meta["wdMAM"] = np.nan_to_num(
            usekernel(
                sub_meta["dMAM"].values, config["min_diff_mam"], config["max_diff_mam"]
            ),
            nan=0.5,
        )
        return sub_meta
    except Exception as e:
        logging.error("Error in computing MAM weights: %s", e)
        raise


# @timing_decorator
def compute_map_weights(
    sub_meta: pd.DataFrame, map0: float, config: Dict[str, Any]
) -> pd.DataFrame:
    """Compute MAP differences and weights."""
    sub_meta["dMAP"] = (
        200 * np.abs(sub_meta["prismMAP"] - map0) / (sub_meta["prismMAP"] + map0)
    )
    sub_meta["wdMAP"] = np.nan_to_num(
        usekernel(
            sub_meta["dMAP"].values, config["min_diff_map"], config["max_diff_map"]
        ),
        nan=0.5,
    )
    return sub_meta


# @timing_decorator
def compute_elevation_weights(
    sub_meta: pd.DataFrame, grd_elev0: float, config: Dict[str, Any]
) -> pd.DataFrame:
    """Compute elevation differences and weights."""
    sub_meta["dElev"] = np.abs(sub_meta["elev_DEM"] - grd_elev0)
    sub_meta["wdElev"] = np.nan_to_num(
        usekernel(
            sub_meta["dElev"].values, config["min_diff_elev"], config["max_diff_elev"]
        ),
        nan=0.5,
    )
    return sub_meta


# @timing_decorator
def compute_dist2coast_weights(
    sub_meta: pd.DataFrame, dist2coast0: float, config: Dict[str, Any]
) -> pd.DataFrame:
    """Compute distance-to-coast differences and weights."""
    sub_meta["dDist2Coast"] = np.abs(sub_meta["dist2coast"] - dist2coast0) / max(
        dist2coast0, 1
    )
    sub_meta["wdDist2Coast"] = np.nan_to_num(
        usekernel(
            sub_meta["dDist2Coast"].values,
            config["min_diff_dist2coast"],
            config["max_diff_dist2coast"],
        ),
        nan=0.5,
    )
    return sub_meta


# @timing_decorator
def compute_elevation_range_and_obstacle_height_weights(
    sub_meta: pd.DataFrame, config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Compute elevation range (ElevR) and obstacle height (ElevOH) differences and
    weights.

    Args:
        sub_meta (pd.DataFrame): DataFrame containing neighbor metadata.
        config (Dict[str, Any]): Configuration dictionary.
    Returns:
        Tuple[pd.DataFrame, Any]: Updated metadata with ElevR/ElevOH differences and
        weights, and the elevation dataset.
    """
    sub_meta["wElevR"] = np.nan_to_num(
        usekernel(
            sub_meta["ElevR"].values, config["min_elev_range"], config["max_elev_range"]
        ),
        nan=0.5,
    )
    sub_meta["wElevOH"] = np.nan_to_num(
        usekernel(
            sub_meta["ElevOH"].values,
            config["min_obstacle_height"],
            config["max_obstacle_height"],
        ),
        nan=0.5,
    )
    sub_meta["wdElevTotal"] = np.power(
        sub_meta["wdElev"] * sub_meta["wElevR"] * sub_meta["wElevOH"], 1 / 3.0
    )
    return sub_meta


# @timing_decorator
def compute_two_sample_tests(
    sub_meta: pd.DataFrame, df_ams: pd.DataFrame, config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Compute two-sample statistical tests for each duration.

    This function updates the `sub_meta` DataFrame in place with the results of
    two-sample statistical tests between the center station and each neighbor station.

    Parameters
    ----------
    sub_meta : pd.DataFrame
        Metadata for stations.
    df_ams : pd.DataFrame
        Annual maximum series data.
    config : Dict[str, Any]
        Configuration parameters.

    Returns
    -------
    pd.DataFrame
        Updated `sub_meta` DataFrame with test results.

    Raises
    ------
    ValueError
        If required configuration parameters are missing or invalid.
    NotImplementedError
        If normalization for the specified `nmodel` is not implemented.
    Exception
        If any other error occurs during computation.
    """
    try:
        # Initialize columns in sub_meta
        sub_meta["wp2ST"] = 1.0  # Initialize weight
        sub_meta["log10p2ST"] = 0.0  # Initialize log10(1/1)=0
        sub_meta["p2ST_KS"] = 1.0  # Initialize p-value for KS test
        sub_meta["p2ST_Chi2"] = 1.0  # Initialize p-value for Chi-squared test
    except Exception as e:
        logger.exception("Error initializing sub_meta columns: %s", e)
        raise

    try:
        # Check conditions to proceed with computation
        ok_compute_2st = config.get("compute_2sample_tests", False)
        nams_min_2st = config.get("nams_min_2ST", 0)

        if not ok_compute_2st:
            logger.info(
                "compute_2sample_tests is False. Skipping two-sample statistical tests."
            )
            return sub_meta

        if "d" not in sub_meta.columns or "nAMS" not in sub_meta.columns:
            raise ValueError("sub_meta must contain 'd' and 'nAMS' columns.")

        if sub_meta["d"].values[0] != 0.0:
            logger.info(
                "First station distance is not zero. Skipping two-sample statistical"
                "tests."
            )
            return sub_meta

        if sub_meta["nAMS"].values[0] < nams_min_2st:
            logger.info(
                "Center station nAMS (%s) is less than nams_min_2ST (%s). "
                "Skipping tests.",
                sub_meta["nAMS"].values[0],
                nams_min_2st,
            )
            return sub_meta
    except Exception as e:
        logger.exception("Error checking conditions for two-sample tests: %s", e)
        raise

    try:
        # Compute normalized precipitation
        sub_ams = df_ams[df_ams["id_num"].isin(sub_meta.index)].copy()
        nmodel = config.get("nmodel")

        if nmodel == 0:
            sub_ams["x"] = sub_ams["precip"]
        elif nmodel == 2:
            if "usedMAM" not in sub_ams.columns:
                raise ValueError("'usedMAM' column is required in df_ams for nmodel=2.")
            sub_ams["x"] = sub_ams["precip"] / sub_ams["usedMAM"]
        elif nmodel == 3:
            if "prismMAP" not in sub_ams.columns:
                raise ValueError("'prismMAP' column is required in df_ams for nmodel=3.")
            sub_ams["x"] = sub_ams["precip"] / sub_ams["prismMAP"]
        else:
            raise NotImplementedError(
                f"Normalization for nmodel={nmodel} is not implemented."
            )
    except Exception as e:
        logger.exception("Error computing normalized precipitation: %s", e)
        raise

    try:
        # Extract center station data
        center_station_id = sub_meta.index[0]
        sub_ams1 = sub_ams[sub_ams["id_num"] == center_station_id].copy()
        if sub_ams1.empty:
            raise ValueError(
                f"No AMS data found for center station ID {center_station_id}."
            )
    except Exception as e:
        logger.exception("Error extracting center station data: %s", e)
        raise

    try:
        # Loop over neighbor stations
        for kn in range(1, len(sub_meta.index)):
            neighbor_station_id = sub_meta.index[kn]
            nams_neighbor = sub_meta["nAMS"].values[kn]

            if nams_neighbor >= nams_min_2st:
                sub_ams2 = sub_ams[sub_ams["id_num"] == neighbor_station_id].copy()
                if not sub_ams2.empty:
                    x1 = sub_ams1["x"].values
                    x2 = sub_ams2["x"].values

                    # Perform Kolmogorov-Smirnov test
                    try:
                        ok_ks_rejected, ks_pvalue = get_kstest_2samp(x1, x2)
                        sub_meta.at[neighbor_station_id, "p2ST_KS"] = ks_pvalue
                    except Exception as e:
                        logger.exception(
                            "Error performing KS test for station %s: %s",
                            neighbor_station_id,
                            e,
                        )
                        sub_meta.at[neighbor_station_id, "p2ST_KS"] = np.nan

                    # Perform Chi-squared test
                    try:
                        ok_chi2_rejected, chi2_pvalue, _ = get_chi2test_2samp_all(x1, x2)
                        sub_meta.at[neighbor_station_id, "p2ST_Chi2"] = chi2_pvalue
                    except Exception as e:
                        logger.exception(
                            "Error performing Chi-squared test for station %s: %s",
                            neighbor_station_id,
                            e,
                        )
                        sub_meta.at[neighbor_station_id, "p2ST_Chi2"] = np.nan

                    # Compute log10p2ST
                    try:
                        p_combined = (ks_pvalue + chi2_pvalue) / 2 + 1e-5
                        log10p2st = np.log10(1 / p_combined)
                        sub_meta.at[neighbor_station_id, "log10p2ST"] = log10p2st
                    except Exception as e:
                        logger.exception(
                            "Error computing log10p2ST for station %s: %s",
                            neighbor_station_id,
                            e,
                        )
                        sub_meta.at[neighbor_station_id, "log10p2ST"] = np.nan
                else:
                    logger.warning(
                        "No AMS data found for neighbor station ID %s.",
                        neighbor_station_id,
                    )
            else:
                logger.info(
                    "Skipping neighbor station ID %s due to insufficient nAMS (%s).",
                    neighbor_station_id,
                    nams_neighbor,
                )
    except Exception as e:
        logger.exception("Error in loop over neighbor stations: %s", e)
        raise

    try:
        # Compute wp2ST
        min_log10p2st = config.get("min_log10p2st")
        max_log10p2st = config.get("max_log10p2st")

        if min_log10p2st is None or max_log10p2st is None:
            raise ValueError(
                "min_log10p2st and max_log10p2st must be provided in config."
            )

        log10p2st_values = sub_meta["log10p2ST"].values
        wp2st_values = usekernel(log10p2st_values, min_log10p2st, max_log10p2st)
        sub_meta["wp2ST"] = np.nan_to_num(wp2st_values, nan=0.5)
    except Exception as e:
        logger.exception("Error computing wp2ST: %s", e)
        raise

    return sub_meta


def combine_weights(sub_meta: pd.DataFrame, config: Dict[str, Any]) -> np.ndarray:
    """Combine weights from various criteria."""
    weights = (
        sub_meta["wd"] * sub_meta["wdMAM"] * sub_meta["wdElevTotal"] * sub_meta["wp2ST"]
    )
    if config.get("use_dist2coast"):
        weights *= sub_meta["wdDist2Coast"]
    if (
        config.get("use_hourly_diff_map_weights")
        or (config.get("ams_duration") == "24h")
        or (config.get("ams_duration") == "10d")
    ):
        weights *= sub_meta["wdMAP"]
    return weights


def combine_parquet_files(
    file_list: List[str], output_file: str, save_as_csv: bool = True
) -> None:
    """
    Combine multiple Parquet files into a single file
    Parameters:
    -----------
    file_list : List[str]
        List of Parquet files to combine
    output_file : str
        Path to the output combined file
    save_as_csv : bool, optional
        If True, saves as CSV. If False, saves as Parquet (default)
    """
    # Read and combine all tables
    tables = [pq.read_table(f) for f in file_list]
    combined_table = pa.concat_tables(tables)

    if save_as_csv:
        # Convert to pandas and save as CSV
        df = combined_table.to_pandas()
        # Remove .parquet extension if present and add .csv
        csv_file = f"{output_file}.csv"
        df.to_csv(csv_file, index=False)
    else:
        # Write combined table as parquet with compression
        pq.write_table(
            combined_table, f"{output_file}.parquet", compression="snappy", index=True
        )
