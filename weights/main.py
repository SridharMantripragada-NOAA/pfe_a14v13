import logging
import os
import time
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import xarray as xr

from .functions import (
    combine_parquet_files,
    combine_weights,
    compute_dist2coast_weights,
    compute_distance_weights,
    compute_distances,
    compute_elevation_range_and_obstacle_height,
    compute_elevation_range_and_obstacle_height_weights,
    compute_elevation_weights,
    compute_mam_weights,
    compute_map_weights,
    compute_two_sample_tests,
    filter_neighbor_stations,
)

logger = logging.getLogger(__name__)


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} execution time: {end_time - start_time:.6f} seconds")
        return result

    return wrapper


# @timing_decorator
def compute_weights_point(
    k: int,
    df_grid: pd.DataFrame,
    df_meta: pd.DataFrame,
    df_ams: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: str,
    elev_srtm=xr.Dataset,
) -> str:
    """
    Process a single grid point and save results to a Parquet file

    Parameters:
    -----------
    k : int
        Grid point index
    df_grid : pd.DataFrame
        Grid data
    df_meta : pd.DataFrame
        Metadata
    df_ams : pd.DataFrame
        Annual max series data
    config : dict
        Configuration dictionary
    output_dir : str
        Directory to save individual Parquet files
    elev_srtm : xr.Dataset
        Elevation data

    Returns:
    --------
    str
        Path to the saved Parquet file
    """
    # Extract grid point details

    id_num0 = df_grid.at[k, "id_num"]
    mam0 = df_grid.at[k, "usedMAM"]
    if config.get("solve_at_station", False) and (id_num0 in df_meta.index):
        mam0 = df_meta.at[id_num0, "usedMAM"]

    map0 = df_grid["prismMAP"].values[k]
    dist2coast0 = (
        df_grid["dist2coast"].values[k] if config.get("use_dist2coast", False) else 0
    )
    src_lat = df_grid["LAT"].values[k]
    src_lon = df_grid["LON"].values[k]
    grid_elev0 = df_grid["elev_DEM"].values[k]

    # Create copy of metadata
    sub_meta = df_meta.copy()
    # Compute weights
    sub_meta["d"] = compute_distances(src_lat, src_lon, sub_meta)
    sub_meta = filter_neighbor_stations(sub_meta, config)
    sub_meta["wd"] = compute_distance_weights(sub_meta["d"], config)
    sub_meta = compute_mam_weights(sub_meta, config, mam0)
    sub_meta = compute_map_weights(sub_meta, map0, config)
    sub_meta = compute_elevation_weights(sub_meta, grid_elev0, config)

    if config.get("compute_obstacle_height", False):
        sub_meta["ElevR"], sub_meta["ElevOH"] = (
            compute_elevation_range_and_obstacle_height(config, sub_meta, elev_srtm)
        )
        sub_meta = compute_elevation_range_and_obstacle_height_weights(sub_meta, config)
    else:
        sub_meta["wdElevTotal"] = sub_meta.get("wdElev", 0.5)

    if config.get("use_dist2coast", False):
        sub_meta = compute_dist2coast_weights(sub_meta, dist2coast0, config)

    sub_meta = compute_two_sample_tests(sub_meta, df_ams, config)
    sub_meta["w"] = combine_weights(sub_meta, config)
    sub_meta["k"] = k
    sub_meta.reset_index(inplace=True)
    # Save to Parquet with compression
    output_file = os.path.join(output_dir, f"grid_point_{k:06d}.parquet")
    sub_meta.to_parquet(output_file, compression="snappy", index=False)

    return sub_meta, output_file


# @timing_decorator
def compute_weights_grid_parallel(
    df_grid: pd.DataFrame,
    df_meta: pd.DataFrame,
    df_ams: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: str,
    final_output: str,
    elev_srtm: xr.DataArray,
    n_jobs: int = -1,
    batch_size: int = 100,
    cleanup_intermediate_files: bool = True,
) -> None:
    """
    Process grid points in parallel and combine results

    Parameters:
    -----------
    df_grid : pd.DataFrame
        Grid data
    df_meta : pd.DataFrame
        Metadata
    df_ams : pd.DataFrame
        Annual max series data
    config : dict
        Configuration dictionary
    output_dir : str
        Directory for intermediate files
    final_output : str
        Path to final combined file
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
    batch_size : int
        Number of grid points to process in each batch
    elev_srtm : optional
        Elevation data
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process grid points in parallel
    total_points = len(df_grid)
    processed_points = 0
    start_time = time.time()

    with joblib.Parallel(n_jobs=n_jobs, verbose=0) as parallel:
        for batch_start in range(0, total_points, batch_size):
            batch_end = min(batch_start + batch_size, total_points)
            parallel(
                joblib.delayed(compute_weights_point)(
                    k, df_grid, df_meta, df_ams, config, output_dir, elev_srtm
                )
                for k in range(batch_start, batch_end)
            )

            # Update progress and elapsed time on same line
            processed_points += batch_end - batch_start
            elapsed_time = time.time() - start_time
            hours, rem = divmod(int(elapsed_time), 3600)
            minutes, seconds = divmod(rem, 60)
            print(
                f"\rProcessed {processed_points} out of {total_points} points "
                f"({(processed_points/total_points)*100:.1f}%) - "
                f"Elapsed time (hh:mm:ss): {hours:02d}:{minutes:02d}:{seconds:02d}",
                end="",
            )

    # Get all generated Parquet files
    all_files = sorted(
        [
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if f.endswith(".parquet")
        ]
    )

    # Combine all files
    combine_parquet_files(all_files, final_output, save_as_csv=True)

    # Optional: Clean up intermediate files
    if cleanup_intermediate_files:
        for f in all_files:
            os.remove(f)


def combine_and_save_weights(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    base_duration: str,
    sdur1: str,
    sdur2: str,
    config: Dict,
) -> pd.DataFrame:
    """
    Combines neighbor weights from two dataframes after checks.

    Parameters:
    -----------
    df1 : pd.DataFrame
        First duration neighbor weights dataframe
    df2 : pd.DataFrame
        Second duration neighbor weights dataframe
    base_duration: str
        daily or sub_daily
    sdur1 : str
        Duration identifier for first dataframe
    sdur2 : str
        Duration identifier for second dataframe
    config : dict
        Options dictionary containing 'arithmetic_mean_weights' and 'proposedOutputPath'

    Returns:
    --------
    Optional[pd.DataFrame]
        Combined neighbor weights if validation passes, None otherwise

    Raises:
    -------
    ValueError
        If input dataframes are empty or have invalid structure
    KeyError
        If required columns are missing
    """
    try:
        logger.info(f"Starting weight combination for durations {sdur1} and {sdur2}")

        # Input validation
        if df1.empty or df2.empty:
            raise ValueError("Input dataframes cannot be empty")

        required_columns = ["id_num", "k", "d", "w"]
        for df, name in [(df1, "df1"), (df2, "df2")]:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise KeyError(f"Missing required columns {missing_cols} in {name}")

        # Check lengths
        nl = len(df1) - len(df2)
        if nl != 0:
            logger.error(
                "Length mismatch for"
                f"durations {sdur1} and {sdur2}: {len(df1)} != {len(df2)}"
            )
            return None

        # Check id_num values
        try:
            ni = (df1.index.values != df2.index.values).sum()
            if ni > 0:
                logger.error(f"Number of different id_num values: {ni}")
                return None
        except Exception as e:
            logger.error(f"Error comparing index values: {str(e)}")
            return None

        # Check k values
        try:
            nk = (df1.k.values != df2.k.values).sum()
            if nk > 0:
                logger.error(f"Number of different k values: {nk}")
                return None
        except Exception as e:
            logger.error(f"Error comparing k values: {str(e)}")
            return None

        logger.info("All validation checks passed. Creating combined dataframe")

        # Create combined dataframe
        try:
            df = pd.DataFrame(
                {
                    "id_num": df1["id_num"],
                    "k": df1["k"],
                    "d": df1["d"],
                    f"w_{sdur1}": df1["w"],
                    f"w_{sdur2}": df2["w"],
                }
            )
        except Exception as e:
            logger.error(f"Error creating combined dataframe: {str(e)}")
            return None

        # Combine weights
        try:
            if config.get("arithmetic_mean_weights", False):
                logger.info("Using arithmetic mean for weight combination")
                df["w"] = (df1["w"].values + df2["w"].values) / 2
            else:
                logger.info("Using geometric mean for weight combination")
                df["w"] = np.sqrt(df1["w"].values * df2["w"].values)
        except Exception as e:
            logger.error(f"Error computing combined weights: {str(e)}")
            return None

        # Reset index and determine duration
        # df = df.reset_index()
        duration = "daily" if base_duration.lower() == "daily" else "subdaily"

        # Save to file
        try:
            output_path = Path(config["proposedOutputPath"])
            output_file = output_path / f"NeighborWeights_{duration}.csv"

            # Ensure directory exists
            output_path.mkdir(parents=True, exist_ok=True)

            df.to_csv(output_file, mode="w", header=True, index=False)
            logger.info(f"Successfully saved combined weights to {output_file}")
        except Exception as e:
            logger.error(f"Error saving combined weights to file: {str(e)}")
            raise

        return df

    except Exception as e:
        logger.error(f"Unexpected error in combine_and_save_weights: {str(e)}")
        raise
