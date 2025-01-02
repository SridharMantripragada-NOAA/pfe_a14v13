import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from .utils import (
    backup_existing_file,
    det_sv_lim_dur,
    find_output_dir,
    get_cov_from_co2,
    get_cov_from_year,
    get_log_filename,
    get_map_extent,
    get_metadata_filename,
    get_param_perc,
)

logger = logging.getLogger(__name__)


def get_additional_config_parameters(
    config: Dict[str, Any], ok_save_log: bool = True
) -> Dict[str, Any]:
    """
    Main function to evaluate parameters and save configuration.

    Args:
        config: Configuration dictionary containing model parameters.
        ok_save_log: Whether to save the log file.

    Returns:
        Updated configuration.
    """
    output_base_path = Path(config["output_base_path"])

    # Set basic paths and files
    config["dir_data1"] = config["prepared_data_region"][config["region"]]
    config["csvInputFileName"] = get_metadata_filename(config)

    # Calculate limits if using generalized weighting
    if config["use_weights_minmax_perc"]:
        config = det_sv_lim_dur(config, True)  # Minimum limits
        config = det_sv_lim_dur(config, False)  # Maximum limits

    # Set covariate flags
    config["okUseCov1"] = config["nmodel"] >= 4
    config["okUseCov2"] = config["ncov2_option"] > -1

    # Determine MAM and MAP covariate usage
    config["okUseCovMAM"] = config["nmodel"] in [1, 2, 21, 22] or (
        config["nmodel"] == 4 and config["ncov1_option"] == 1
    )
    config["okUseCovMAP"] = (
        config["nmodel"] == 3
        or (config["okUseCov1"] and config["ncov1_option"] == 1)
        or (config["nmodel"] == 4 and config["ncov1_option"] == 0)
    )

    # Calculate temporal covariate based on option
    if config["ncov2_option"] == 0:  # CO2-based
        config["cov2_out"] = get_cov_from_co2(
            np.array([config["pre_industrial_co2"], config["current_co2"]]), config
        )
    elif config["ncov2_option"] == 1:  # GWL-based
        config["cov2_out"] = np.array(
            [config["pre_industrial_gwl"], config["current_gwl"]]
        )
    elif config["ncov2_option"] == 2:  # Year-based
        config["cov2_out"] = get_cov_from_year(
            np.array([config["base_year"], config["current_year"]]), config
        )
    else:
        config["cov2_out"] = [0.0]

    # Set map extent
    config["mapExtent"] = get_map_extent(config)

    # Create output directories
    try:
        output_base_path.mkdir(exist_ok=True)
    except OSError as e:
        logger.exception("Failed to create output_base_path directory.")
        raise RuntimeError("Failed to create base output directory") from e

    proposed_output_path = output_base_path / find_output_dir(config)
    try:
        proposed_output_path.mkdir(exist_ok=True)
    except OSError as e:
        logger.exception("Failed to create proposed_output_path directory.")
        raise RuntimeError("Failed to create proposed output directory") from e

    duration_path = proposed_output_path / config["ams_duration"]
    try:
        duration_path.mkdir(exist_ok=True)
    except OSError as e:
        logger.exception("Failed to create duration_path directory.")
        raise RuntimeError("Failed to create duration directory") from e

    config["proposedOutputPath"] = str(proposed_output_path)

    # Get parameter percentiles
    config["param_perc"] = get_param_perc(config)

    # Save log if requested
    if ok_save_log:
        logger.info("Proposed output path: %s", config["proposedOutputPath"])

        # Generate log filename
        log_filename = get_log_filename(
            config["csvInputFileName"], config["ams_duration"], config
        )

        # Backup existing file
        backup_existing_file(duration_path, log_filename, ".csv", config)

        try:
            pd.DataFrame([config]).to_csv(duration_path / log_filename, index=False)
        except Exception as e:
            logger.exception("Failed to save configuration to CSV.")
            raise RuntimeError("Failed to save configuration log") from e

    return config
