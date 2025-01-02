import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import rasterio as rs
import xarray as xr

logger = logging.getLogger(__name__)


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} execution time: {end_time - start_time:.6f} seconds")
        return result

    return wrapper


def get_coord_from_src(src: rs.io.DatasetReader) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the coordinate arrays (longitude and latitude) from the source metadata.

    Parameters:
        src (rasterio.io.DatasetReader): Source object containing metadata about the
        PRISM grid.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - lons: Array of longitudes.
            - lats: Array of latitudes, in decreasing order.
    """
    # Compute longitude and latitude steps
    dlon = (src.bounds.right - src.bounds.left) / src.width
    dlat = (src.bounds.top - src.bounds.bottom) / src.height

    # Compute coordinate arrays
    lons = np.arange(src.bounds.left + dlon / 2, src.bounds.right, dlon)
    lats = np.arange(src.bounds.top - dlat / 2, src.bounds.bottom, -dlat)

    return lons, lats


def flip_prism(src, band: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Flips the PRISM grid along the latitude axis to ensure the latitude array is in
    increasing order.

    Parameters:
        src: Source object containing information to compute coordinate arrays.
        band (np.ndarray): 2D array representing the PRISM data grid.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - lons: Array of longitudes.
            - lats: Array of latitudes, flipped to be in increasing order.
            - flipped_band: Data grid flipped along the latitude axis.
    """
    # Compute coordinate arrays from the source information
    lons, lats = get_coord_from_src(src)
    # Use flip_prism1 to flip the latitude array and the data grid
    return flip_prism1(lons, lats, band)


def flip_prism1(
    lons: np.ndarray, lats: np.ndarray, band: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Flips the latitude array and the data grid along the latitude axis.

    Parameters:
        lons (np.ndarray): Array of longitudes.
        lats (np.ndarray): Array of latitudes.
        band (np.ndarray): 2D array representing the PRISM data grid.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - lons: Unchanged array of longitudes.
            - flipped_lats: Latitudes array, flipped to be in increasing order.
            - flipped_band: Data grid flipped along the latitude axis.
    """
    flipped_lats = np.flip(lats)
    flipped_band = np.flip(band, axis=0)

    return lons, flipped_lats, flipped_band


# @timing_decorator
def get_prism_array(
    src,
    band: np.ndarray,
    longitude: List[float],
    latitude: List[float],
    ok_allow_extrapolation: bool = False,
) -> np.ndarray:
    """
    Extract precipitation values from a PRISM grid for a list of latitude and longitude
    points.

    Parameters:
        src: Source object containing metadata for the PRISM grid.
        band (np.ndarray): 2D array representing the PRISM data grid.
        longitude (List[float]): List of longitude values for which to extract
        precipitation.
        latitude (List[float]): List of latitude values for which to extract
        precipitation.
        ok_allow_extrapolation (bool): Flag to allow extrapolation if the points are
        outside the grid boundaries. Default is False.

    Returns:
        np.ndarray: Array of precipitation values for the specified latitudes and
        longitudes.
    """
    # Flip the PRISM grid along the latitude axis to ensure latitudes are in increasing
    # order
    lons, lats, bandr = flip_prism(src, band)

    # Extract precipitation values for the given longitude and latitude points
    return get_prism_array1(
        bandr, lons, lats, longitude, latitude, ok_allow_extrapolation
    )


# @timing_decorator
def read_prism_raster(
    filepath: str, file_name: str, exit_if_failed: bool = True
) -> Tuple[Union[rs.io.DatasetReader, np.ndarray], np.ndarray]:
    """
    Reads a raster file containing precipitation data in the PRISM grid format.

    Parameters:
        filepath (str): Directory path where the raster file is located.
        file_name (str): File name of the raster to read.
        exit_if_failed (bool): Flag to determine if the program should exit if the file
        is not found. Default is True.

    Returns:
        Tuple[Union[rs.io.DatasetReader, np.ndarray], np.ndarray]:
            - src: Rasterio DatasetReader object or an empty array if the file is not
            found.
            - band: 2D array of precipitation data with no-data values replaced by NaN,
            or an empty array if the file is not found.
    """
    try:
        # Construct the full file path
        file_path = Path(filepath, file_name)

        # Check if the file exists
        if file_path.exists():
            src = rs.open(file_path)
            band = src.read(1)

            # Replace no-data values and negative values with NaN
            band = np.where(band == src.nodata, np.nan, band)
            band = np.where(band < 0, np.nan, band)

        else:
            if exit_if_failed:
                logging.critical(
                    f"Error: File {file_path} was not found. Exiting execution."
                )
                sys.exit()
            else:
                logging.warning(
                    f"Warning: File {file_path} was not found. Returning empty arrays."
                )
                return np.array([]), np.array([])

        return src, band

    except Exception as e:
        logging.error(f"Error reading PRISM raster: {e}")
        return np.array([]), np.array([])


# @timing_decorator
def get_a14_precip(
    dir_data: str, df_grid_in: pd.DataFrame, sdur: str, region: str
) -> pd.DataFrame:
    """
    Extract A14 precipitation arrays for a list of latitude and longitude values.

    Parameters:
        dir_data (str): Directory path containing the A14 precipitation raster files.
        df_grid_in (pd.DataFrame): Input DataFrame containing latitude and longitude
        columns.
        sdur (str): Duration string for the precipitation data (e.g., '24h').
        region (str): Version identifier (e.g., 'V12').

    Returns:
        pd.DataFrame: Updated DataFrame with extracted A14 precipitation values for
        specified return periods.
    """
    # Desired return periods for A14 precipitation data
    return_periods = [2, 25, 100]

    for period in return_periods:
        # Determine the string representation of the return period
        str_return_period = "2.54y" if period == 2 else f"{period}y"
        s_grid_col = f"A14p_{str_return_period}"  # Column label for A14 precip data

        # Check if the column already exists, and if not, initialize it
        if s_grid_col not in df_grid_in.columns:
            df_grid_in[s_grid_col] = np.nan  # Initialize the new column with NaN

            # Read the A14 precipitation raster for the specified return period
            src_a14_precip, band_a14_precip = read_a14_raster(
                dir_data, sdur, period, region
            )

            if (
                len(band_a14_precip) > 0
            ):  # Check if the raster file was successfully loaded
                # Extract precipitation values from the raster and populate the DataFrame
                df_grid_in[s_grid_col] = get_prism_array(
                    src_a14_precip,
                    band_a14_precip,
                    df_grid_in.LON.values,
                    df_grid_in.LAT.values,
                )

    return df_grid_in


# @timing_decorator
def read_a14_raster(
    filepath: str, sdur: str, return_period: int, region: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read the A14 precipitation grid for a specified duration and return period.

    Parameters:
        filepath (str): Path to the directory containing the A14 precipitation raster
        files.
        sdur (str): Duration string (e.g., '60m', '06h', '24h', '04d', '10d', '60d').
        return_period (int): Return period (in years) for the precipitation data.
        region (str): Version identifier ('V11', 'V12', or 'conus').

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - src_a14_precip: The source object or array from the PRISM raster.
            - band_a14_precip: The A14 precipitation data, converted to inches if
            necessary.
    """
    # List of available unconstrained durations for A14 precipitation grid
    valid_durations = ["60m", "06h", "24h", "04d", "10d", "60d"]
    if sdur not in valid_durations:
        print(
            f"  Warning: in read_a14_raster, duration label '{sdur}' "
            "is not in the list of base durations."
        )

    # Set the conversion factor to inches
    fc = 1.0  # Default conversion factor (no conversion)

    # Construct the file name based on the version identifier
    if region == "V11":
        str_return_period = f"{return_period}yr"
        file_name = f"tx{str_return_period}{sdur}a.asc"
        # Uncomment the following line if a different conversion factor is needed for V11
        # fc = 0.01 / 25.4
    elif region == "V12":
        str_return_period = f"{str(return_period).zfill(4)}y"
        file_name = f"vol12_{sdur}_{str_return_period}.asc"
        # Uncomment the following line if a different conversion factor is needed for V12
        # fc = 1 / 25.4
    elif region == "conus":
        str_return_period = f"{return_period}yr"
        file_name = f"A14_{str_return_period}{sdur}.asc"
        fc = 1.0 / 1000  # Conversion factor to inches
    else:
        file_name = "file_does_not_exist.asc"

    # Read the A14 precipitation raster
    # Note: `read_prism_raster` should be a defined function that reads the raster file
    src_a14_precip, band_a14_precip = read_prism_raster(
        filepath, file_name, exit_if_failed=False
    )

    # Apply the conversion factor to the precipitation data
    return src_a14_precip, band_a14_precip * fc


# @timing_decorator
def prepare_grid_data(
    config: dict,
    df_grid: pd.DataFrame,
    df_meta: pd.DataFrame,
    ds_elev: xr.Dataset,
    ds_prism_map: xr.Dataset,
) -> pd.DataFrame:
    """
    Prepare grid data for distribution fitting, including MAP, MAM, elevation, and
    distance to coast.
    """
    try:
        if "prism" in df_grid.columns:
            df_grid = df_grid.rename(columns={"prism": "prismMAP"})
        elif "prismMAP" not in df_grid.columns:
            if ds_prism_map is None:
                data_dir = Path(config["dir_data0"])
                prism_path = data_dir / config["prism_file"]
                logger.info(
                    f"Loading PRISM MAP data from {prism_path} for grid points..."
                )
                ds_prism_map = xr.open_dataset(prism_path, engine="netcdf4")
            df_grid["pid"] = np.arange(len(df_grid))
            df_grid["prismMAP"] = get_prism_array1(
                ds_prism_map.pr.values[0],
                ds_prism_map.longitude.values,
                ds_prism_map.latitude.values,
                df_grid["LON"].values,
                df_grid["LAT"].values,
            )
            df_grid["prismMAP"] = df_grid["prismMAP"].round(2)

        if config.get("use_dist2coast", False) and "dist2coast" not in df_grid.columns:
            data_dir = Path(config["dir_data0"])
            dist2coast_file = data_dir / config["dist2coast_file"]
            logger.info(
                f"Loading distance to coast data from {dist2coast_file} for "
                "grid points..."
            )
            ds_dist2coast = xr.open_dataset(dist2coast_file)
            logger.info("Extracting dist2coast values for grid points...")
            df_grid["dist2coast"] = get_prism_array1(
                ds_dist2coast.dist.values,
                ds_dist2coast.lon.values,
                ds_dist2coast.lat.values,
                df_grid["LON"].values,
                df_grid["LAT"].values,
            )
            df_grid["dist2coast"] = df_grid["dist2coast"].fillna(0).round(1).clip(lower=0)

        if config.get("solve_at_station", False):
            if "gridMAM" not in df_grid.columns:
                logger.info("Merging gridMAM from df_meta into df_grid...")
                df_grid = df_grid.merge(
                    df_meta[["gridMAM"]], how="left", left_on="id_num", right_index=True
                )
            if "meanAMS" not in df_grid.columns:
                logger.info("Merging meanAMS from df_meta into df_grid...")
                df_grid = df_grid.merge(
                    df_meta[["meanAMS"]], how="left", left_on="id_num", right_index=True
                )
            if "usedMAM" not in df_grid.columns:
                logger.info("Merging usedMAM from df_meta into df_grid...")
                df_grid = df_grid.merge(
                    df_meta[["usedMAM"]], how="left", left_on="id_num", right_index=True
                )
        else:
            if "gridMAM" not in df_grid.columns:
                logger.info("Loading MAM grid data for grid points...")
                lons_mam, lats_mam, array_mam = read_mam_raster(
                    config["dir_data1"],
                    config["ams_duration"],
                    config["region"],
                    config.get("use_greg_mam", False),
                )
                logger.info("Extracting gridMAM values for grid points...")
                df_grid["gridMAM"] = np.nan
                indices = df_grid.index
                df_grid.loc[indices, "gridMAM"] = get_prism_array1(
                    array_mam,
                    lons_mam,
                    lats_mam,
                    df_grid.loc[indices, "LON"].values,
                    df_grid.loc[indices, "LAT"].values,
                )
            if "usedMAM" not in df_grid.columns:
                df_grid["usedMAM"] = df_grid["gridMAM"]
                nan_count = df_grid["usedMAM"].isna().sum()
                if nan_count > 0:
                    logger.warning(f"{nan_count} grid points have NaN usedMAM values.")

        if "elev_DEM" not in df_grid.columns:
            logger.info("Extracting elevation data for grid points...")
            df_grid["elev_DEM"] = np.nan
            indices = df_grid.index
            df_grid.loc[indices, "elev_DEM"], ds_elev = read_elevation(
                config["region"],
                config["dir_data1"],
                df_grid.loc[indices, "LAT"],
                df_grid.loc[indices, "LON"],
                ds_elev=ds_elev,
            )

        if config.get("output_a14_precip", False):
            logger.info("Including A14 precipitation estimates...")
            df_grid = get_a14_precip(
                config["dir_data1"],
                df_grid.copy(),
                config["ams_duration"],
                config["region"],
            )
        return df_grid
    except Exception as e:
        logger.error(f"Failed to prepare grid data: {e}")
        raise


# @timing_decorator
def read_mam_raster(
    filepath: str, sdur: str, region: str, use_greg_mam: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads the PRISM mean annual maximum (MAM) precipitation grid.

    Parameters:
        filepath (str): Path to the directory containing the MAM raster files.
        sdur (str): Duration for which the MAM grid is required (e.g., '60m', '24h').
        region (str): Version identifier ('V11' or 'V12').
        use_greg_mam (bool): Flag to use Greg's MAM final grid. Default is False.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - lons_mam: Array of longitudes.
            - lats_mam: Array of latitudes.
            - array_mam: MAM precipitation data, converted to inches if applicable.
    """
    try:
        if use_greg_mam:
            # Use Greg's MAM final grid
            fc = 1.0
            if sdur == "24h":
                s = "1_day"
            elif sdur == "60m":
                s = "1_hour"
            else:
                logging.warning(
                    f"Greg MAM grid not available for duration: {sdur}. "
                    "Defaulting to '1_day'."
                )
                s = "1_day"
                fc = np.nan  # Return NaN values in the grid

            if region == "V12":
                spath = Path(
                    filepath,
                    f"MAM_vol12_{s}_0.8_km_regression_and_bias_kriging_result.nc",
                )
            else:
                spath = Path(
                    filepath,
                    f"MAM_conus_{s}_0.8_km_regression_and_diff_kriging_result.nc",
                )

            # Open the NetCDF dataset
            ds = xr.open_dataset(spath)
            lons_mam = ds.longitude.values
            lats_mam = np.flip(ds.latitude.values)
            array_mam = np.flip(ds.Data.values, axis=0) * fc

        else:
            # Use PRISM MAM
            conversion_factors = {
                "V11": 0.01 / 25.4,
                "V12": 1 / 25.4,
            }  # Move this to configuration YAML
            if region not in conversion_factors:
                logging.error(f"Prism MAM grid not available for version: {region}")
                sys.exit()

            if region == "V11" and sdur == "60m":
                sdur = "01h"
            file_name = (
                f"tx_ppt_{sdur}_mam_01262018.arc"
                if region == "V11"
                else f"idmtwy_ppt_{sdur}_mam.tif"
            )
            fc = conversion_factors[region]

            # Read the PRISM raster
            src_mam, band_mam = read_prism_raster(filepath, file_name)
            if len(band_mam) == 0:
                logging.error(f"Error: MAM array not read from {filepath}")
                return np.array([]), np.array([]), np.array([])

            lons_mam, lats_mam, array_mam = flip_prism(src_mam, band_mam)
            array_mam *= fc  # Convert to inches

        return lons_mam, lats_mam, array_mam
    except Exception as e:
        logger.error(f"Failed to prepare grid data: {e}")
        raise


# @timing_decorator
def get_prism_array1(
    precip: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    longitude: List[float],
    latitude: List[float],
    ok_allow_extrapolation: bool = False,
) -> np.ndarray:
    """
    Extracts precipitation values from a gridded precipitation array for given
    longitude and latitude coordinates.

    Parameters:
        precip (np.ndarray): 2D array of precipitation data.
        lons (np.ndarray): 1D array of longitudes corresponding to the columns of
        the precip array.
        lats (np.ndarray): 1D array of latitudes corresponding to the rows of the
        precip array.
        longitude (List[float]): List of longitude values for extraction.
        latitude (List[float]): List of latitude values for extraction.
        ok_allow_extrapolation (bool): Flag to allow or disallow extrapolation.
        Default is False.

    Returns:
        np.ndarray: Array of precipitation values, with NaN for missing or
        out-of-bound data.
    """
    try:
        # Initialize list for precipitation values
        precip_values: List[Union[float, np.nan]] = []

        # Extract precipitation values for each (latitude, longitude) pair
        for lat, lon in zip(latitude, longitude):
            try:
                # Find the nearest indices for the given lat, lon
                col = np.argmin(np.abs(lons - lon))
                row = np.argmin(np.abs(lats - lat))
                precip_value = round(precip[row, col], 3)
                precip_values.append(precip_value)
            except Exception as e:
                logging.error(
                    f"Error extracting precipitation for coordinates ({lat}, {lon}): {e}"
                )
                precip_values.append(np.nan)

        # Convert the list to a NumPy array
        precip_array: np.ndarray = np.array(precip_values, dtype=np.float64)

        # Check for out-of-bound coordinates
        out_of_bounds: np.ndarray = (
            (np.array(longitude) < np.min(lons))
            | (np.array(longitude) > np.max(lons))
            | (np.array(latitude) < np.min(lats))
            | (np.array(latitude) > np.max(lats))
        )
        if np.any(out_of_bounds):
            logging.warning(
                f"{np.sum(out_of_bounds)} points are outside the grid boundaries."
            )
            if not ok_allow_extrapolation:
                precip_array[out_of_bounds] = np.nan

        # Handle values that are too small
        precip_array[precip_array < 0.001] = np.nan

        # Alert for NaN values in the result
        nan_values_count: int = np.isnan(precip_array).sum()
        if nan_values_count > 0:
            logging.warning(
                f"{nan_values_count} NaN values returned out of {len(precip_array)} "
                "total points."
            )

        return precip_array
    except Exception as e:
        logger.error(f"Failed to prepare grid data: {e}")
        raise


# @timing_decorator
def read_metadata(config: dict) -> pd.DataFrame:
    """
    Load station metadata from CSV files specified in the configuration.
    """
    try:
        file_name0 = config["meta_file"]
        file_name = file_name0.replace(".csv", f"_{config['ams_duration']}.csv")
        data_dir = Path(config["dir_data1"])
        metadata_path = data_dir / file_name
        if not metadata_path.exists():
            metadata_path = (
                data_dir / file_name0
            )  # Use default if duration-specific file doesn't exist
        logger.info(f"Reading metadata from {metadata_path}...")
        df_meta = pd.read_csv(metadata_path).set_index("id_num")
        if "HDSC" in df_meta.columns:
            df_meta["HDSC"] = df_meta["HDSC"].str.replace("-", "_")
        return df_meta
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        raise


# @timing_decorator
def read_map(config: dict, df_meta: pd.DataFrame) -> Tuple[xr.Dataset, pd.DataFrame]:
    """
    Prepare Mean Annual Precipitation (MAP) data by loading PRISM datasets and
    extracting MAP values.
    """

    try:
        if "prism" in df_meta.columns:
            df_meta.rename(columns={"prism": "prismMAP"}, inplace=True)
            ds_prism_map = None
        elif "prismMAP" not in df_meta.columns:
            data_dir = Path(config["dir_data0"])
            prism_path = data_dir / config["prism_file"]
            logger.info(f"Loading PRISM MAP data from {prism_path}...")
            ds_prism_map = xr.open_dataset(prism_path, engine="netcdf4")
            logger.info("Extracting prismMAP values for stations...")
            df_meta["prismMAP"] = get_prism_array1(
                ds_prism_map.pr.values[0],
                ds_prism_map.longitude.values,
                ds_prism_map.latitude.values,
                df_meta["LON"].values,
                df_meta["LAT"].values,
            )
        # Filter stations based on MAP coverage, removing those with NaN 'prismMAP'
        # values if required.
        if config.get("okUseCovMAP", False):
            initial_count = len(df_meta)
            df_meta.dropna(subset=["prismMAP"], inplace=True)
            removed_count = initial_count - len(df_meta)
            if removed_count > 0:
                logger.info(
                    f"Removed {removed_count} stations due to missing prismMAP values."
                )

        df_meta["prismMAP"] = df_meta["prismMAP"].round(2)
        return ds_prism_map, df_meta
    except Exception as e:
        logger.error(f"Failed to prepare MAP data: {e}")
        raise


# @timing_decorator
def read_mam(config: dict, df_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare Mean Annual Maximum (MAM) precipitation data by loading MAM grids and
    extracting values.
    """
    try:
        if "gridMAM" not in df_meta.columns:
            logger.info("Loading MAM grid data...")
            lons_mam, lats_mam, array_mam = read_mam_raster(
                config["dir_data1"],
                config["ams_duration"],
                config["region"],
                config.get("use_greg_mam", False),
            )
            logger.info("Extracting gridMAM values for stations...")
            df_meta["gridMAM"] = get_prism_array1(
                array_mam,
                lons_mam,
                lats_mam,
                df_meta["LON"].values,
                df_meta["LAT"].values,
            )
            nan_count = df_meta["gridMAM"].isna().sum()
            if nan_count > 0:
                logger.warning(f"{nan_count} stations have NaN gridMAM values.")
        return df_meta
    except Exception as e:
        logger.error(f"Failed to prepare MAM data: {e}")
        raise


# @timing_decorator
def read_elevation(
    region: str,
    filepath: str,
    latitude: List[float],
    longitude: List[float],
    ds_elev: xr.Dataset = None,
) -> Tuple[np.ndarray, xr.Dataset]:
    """
    Extracts elevation data from an SRTM90m (Shuttle Radar Topography Mission) netCDF
    file for a list of latitudes and longitudes.

    Parameters:
        region (str): Volume
        filepath (str): Directory path
        latitude (List[float]): List of latitude values for the stations.
        longitude (List[float]): List of longitude values for the stations.
        ds_elev (xr.Dataset, optional): Preloaded xarray Dataset containing
        elevation data. Defaults to None.

    Returns:
        Tuple[np.ndarray, xr.Dataset]:
            - Array of elevation values for the given latitude and longitude coordinates.
            - The xarray Dataset used, to avoid reopening the file repeatedly.
    """
    try:
        if ds_elev is None:  # Open the netCDF file if nread_dist2coastot provided
            file_name = f"elevation_SRTM180m_{region.replace('V', 'Vol')}.nc"
            file_path = Path(filepath, file_name)
            ds_elev = xr.open_dataset(file_path, engine="netcdf4")
            logging.info(f"Opened elevation data file: {file_path}")

        # Extract the 'elevation' variable from the Dataset
        elev_srtm = ds_elev["elevation"]

        # Initialize list to store elevation data
        elev_stn_srtm = []

        # Extract elevation data for each coordinate
        for i, (latt, lonn) in enumerate(zip(latitude, longitude), start=1):
            try:
                # Use nearest neighbor search to get elevation
                elevation = elev_srtm.sel(
                    lat=latt, lon=lonn, method="nearest"
                ).values.item()
                elev_stn_srtm.append(elevation)
            except Exception as e:
                logging.error(
                    "Error extracting elevation for coordinates "
                    f"(lat: {latt}, lon: {lonn}): {e}"
                )
                elev_stn_srtm.append(np.nan)  # Use NaN for failed extractions

        return np.array(elev_stn_srtm, dtype=np.float64), ds_elev
    except Exception as e:
        logging.critical(f"Critical error in read_elevation: {e}")


# @timing_decorator
def read_dist2coast(config: dict, df_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare distance to coast data by extracting values for each station.
    """
    try:
        if config.get("use_dist2coast", False) and "dist2coast" not in df_meta.columns:
            data_dir = Path(config["dir_data0"])
            dist2coast_file = data_dir / "dist2coast_1deg_regrid.nc"
            logger.info(f"Loading distance to coast data from {dist2coast_file}...")
            ds_dist2coast = xr.open_dataset(dist2coast_file)
            logger.info("Extracting dist2coast values for stations...")
            df_meta["dist2coast"] = get_prism_array1(
                ds_dist2coast.dist.values,
                ds_dist2coast.lon.values,
                ds_dist2coast.lat.values,
                df_meta["LON"].values,
                df_meta["LAT"].values,
            )
            df_meta["dist2coast"] = df_meta["dist2coast"].fillna(0).round(1).clip(lower=0)
        return df_meta
    except Exception as e:
        logger.error(f"Failed to prepare distance to coast data: {e}")
        raise


# @timing_decorator
def read_grid(config: dict, df_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Load grid points for distribution fitting from the specified CSV file.
    """
    try:
        input_file = Path(config["dir_data1"]) / config["csvInputFileName"]
        logger.info(f"Loading grid points from {input_file}...")
        df_grid = pd.read_csv(input_file)
        if config.get("solve_at_station", False) and "HDSC" in df_grid.columns:
            df_grid["HDSC"] = df_grid["HDSC"].str.replace("-", "_")
        if config.get("okUseCovMAP", False) and config.get("solve_at_station", False):
            initial_count = len(df_grid)
            df_grid = df_grid[df_grid["id_num"].isin(df_meta.index)]
            removed_count = initial_count - len(df_grid)
            if removed_count > 0:
                logger.info(
                    f"Removed {removed_count} grid points due to missing prismMAP values."
                )
        return df_grid
    except Exception as e:
        logger.error(f"Failed to load grid points: {e}")
        raise
