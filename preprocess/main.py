import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import xarray as xr

from .ams_processing import load_stats_to_meta, read_ams
from .covariates_processing import load_covariate_options
from .read_input import (
    prepare_grid_data,
    read_dist2coast,
    read_elevation,
    read_grid,
    read_mam,
    read_map,
    read_metadata,
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
def preprocess_input_data(
    config: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[xr.Dataset]]:
    """
    Main function to orchestrate data preprocessing steps.

    Args:
        config (dict): Configuration dictionary containing options and parameters.

    Returns:
        Tuple containing:
            - df_grid (pd.DataFrame): Prepared grid DataFrame
            - df_meta (pd.DataFrame): Station metadata DataFrame
            - df_ams (pd.DataFrame): Annual maximum series DataFrame
            - ds_elev (Optional[xr.Dataset]): Elevation dataset
    """
    try:
        # Load metadata
        df_meta = read_metadata(config)
        # Load MAP data
        ds_prism_map, df_meta = read_map(config, df_meta)
        # Load MAM data
        df_meta = read_mam(config, df_meta)
        # Load elevation data
        if "elev_DEM" not in df_meta.columns:
            logger.info("Extracting elevation data for stations...")
            elev_values, ds_elev = read_elevation(
                config["region"], config["dir_data1"], df_meta["LAT"], df_meta["LON"]
            )
            df_meta["elev_DEM"] = elev_values
        # Load distance to coast data
        df_meta = read_dist2coast(config, df_meta)
        # Load AMS data
        df_ams = read_ams(config, config["ams_duration"], df_meta)
        # Load stats to metadata
        df_meta, df_ams = load_stats_to_meta(config, df_meta, df_ams)
        # Load covariate options
        df_meta, df_ams = load_covariate_options(config, df_meta, df_ams)
        # Save AMS, metadata, and grid data if required
        # Load and prepare grid data
        df_grid = read_grid(config, df_meta)
        if config.get("save_plots", False):
            save_processed_data(config, df_ams, df_meta)
            save_grid_data(config, df_grid)
        df_grid = prepare_grid_data(config, df_grid, df_meta, ds_elev, ds_prism_map)
        return df_grid, df_meta, df_ams, ds_elev
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        raise


# @timing_decorator
def save_processed_data(
    config: dict, df_ams: pd.DataFrame, df_meta: pd.DataFrame
) -> None:
    """Save AMS and metadata to CSV files."""
    try:
        output_dir = Path(config["proposedOutputPath"]) / config["ams_duration"]
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving df_ams to {output_dir}...")
        df_ams.to_csv(output_dir / f"df_ams_{config['ams_duration']}.csv", index=False)

        logger.info(f"Saving df_meta to {output_dir}...")
        df_meta.to_csv(output_dir / f"df_meta_{config['ams_duration']}.csv")
    except Exception as e:
        logger.error(f"Failed to save processed data: {e}")
        raise


# @timing_decorator
def save_grid_data(config: dict, df_grid: pd.DataFrame) -> None:
    """Save grid data to CSV."""
    try:
        output_dir = Path(config["proposedOutputPath"]) / config["ams_duration"]
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving df_grid to {output_dir}...")
        df_grid.to_csv(output_dir / f"df_grid_{config['ams_duration']}.csv", index=False)
    except Exception as e:
        logger.error(f"Failed to save grid data: {e}")
        raise
