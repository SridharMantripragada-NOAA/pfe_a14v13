from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)


def convert_duration_to_minutes(
    sdur: Union[str, List[str]], config: Dict[str, Any]
) -> Union[int, List[int]]:
    """
    Convert duration strings to minutes.

    Args:
        sdur: Duration string(s) ending with 'h' for hours or 'd' for days.
        config: Configuration dictionary.

    Returns:
        Duration(s) in minutes.
    """

    def convert_single_duration(sd: str) -> int:
        if len(sd) < 2:
            logger.error("Invalid duration format: '%s'", sd)
            raise ValueError("Invalid duration format. Use 'Nh' or 'Nd'")

        value = sd[:-1]
        unit = sd[-1]

        try:
            minutes = int(value)
        except ValueError as e:
            logger.error("Invalid numeric value in duration: '%s'", sd)
            raise ValueError(f"Invalid numeric value in duration: '{sd}'") from e

        if unit == "h":
            minutes *= 60
        elif unit == "d":
            minutes *= 60 * 24
        else:
            logger.error("Invalid duration unit: '%s'", unit)
            raise ValueError("Invalid duration unit. Use 'h' for hours or 'd' for days")

        return minutes

    if isinstance(sdur, str):
        return convert_single_duration(sdur)
    if isinstance(sdur, list):
        return [convert_single_duration(sd) for sd in sdur]
    logger.error("Invalid input type for duration: %s", type(sdur))
    raise TypeError("Input must be a string or list of strings")


def interpolate2d(
    long: Union[float, np.ndarray],
    latg: Union[float, np.ndarray],
    lon2: np.ndarray,
    lat2: np.ndarray,
    z2: np.ndarray,
    config: Dict[str, Any],
) -> np.ndarray:
    """
    Perform 2D interpolation of gridded data.

    Args:
        long: Target longitude points.
        latg: Target latitude points.
        lon2: Source longitude grid.
        lat2: Source latitude grid.
        z2: Source values.
        config: Configuration dictionary.

    Returns:
        Interpolated values as a numpy array.
    """
    try:
        interp = RegularGridInterpolator(
            (lat2, lon2), z2, method="linear", bounds_error=False, fill_value=None
        )
        return interp((latg, long))
    except Exception as e:
        logger.exception("2D interpolation failed")
        raise RuntimeError("Interpolation failed") from e


def get_metadata_filename(config: Dict[str, Any]) -> str:
    """
    Generate input CSV filename based on configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Input CSV filename.
    """
    sv = config["region"]
    if config["solve_at_station"]:
        if sv.startswith("V"):
            return f"MetaData_Atlas14_{sv.replace('V', 'Vol')}_buffer30km_r.csv"
        return "df_ams_meta.csv"
    return f"XYPRISM2_points{sv}_buffer10km.csv"


def det_sv_lim_dur(config: Dict[str, Any], ok_min: bool) -> Dict[str, Any]:
    """
    Calculate SV limits based on duration for the specified region.

    Args:
        config: Configuration dictionary.
        ok_min: True for minimum limits, False for maximum limits.

    Returns:
        Updated configuration with new limits.
    """
    sdur = config["ams_duration"]
    perc = config["wmin_perc"] if ok_min else config["wmax_perc"]
    perc0 = np.array(config["perc0"])

    region = config["region"]
    if region not in ["V12", "V11"]:
        logger.error("Unsupported region: %s", region)
        raise ValueError(f"Unsupported region: {region}. Supported: 'V12', 'V11'.")

    region_config = config["region_specific"][region]
    sdurl = region_config["sdurl"]
    dmap0 = region_config["diff_map0"]
    delev0 = region_config["diff_elev0"]
    elevr0 = region_config["elev_range0"]
    elevoh0 = region_config["obstacle_height0"]
    dmam0 = np.array(region_config["diff_mam0"])
    p2st0 = np.array(region_config["p2st0"])

    log10dur0 = np.log10(convert_duration_to_minutes(sdurl, config))
    log10dur = np.log10(convert_duration_to_minutes(sdur, config))
    dmap = np.interp(perc, perc0, dmap0)
    delev = np.interp(perc, perc0, delev0)
    elevr = np.interp(perc, perc0, elevr0)
    elevoh = np.interp(perc, perc0, elevoh0)
    dmam = interpolate2d(perc, log10dur, perc0, log10dur0, dmam0, config)
    p2st = interpolate2d(perc, log10dur, perc0, log10dur0, p2st0, config)

    if config["apply_cap2limits"]:
        dmam = max(dmam, config["min0_diff_mam"])
        dmap = max(dmap, config["min0_diff_map"])
        delev = max(delev, config["min0_diff_elev"])
        elevr = max(elevr, config["min0_elev_range"])
        elevoh = max(elevoh, config["min0_obstacle_height"])
        p2st = min(max(p2st, config["min0_p2st"]), config["max0_p2st"])

    prefix = "min_" if ok_min else "max_"
    config.update(
        {
            f"{prefix}dMAM": dmam,
            f"{prefix}dMAP": dmap,
            f"{prefix}dElev": delev,
            f"{prefix}elevR": elevr,
            f"{prefix}elevOH": elevoh,
            f"{prefix}log10p2ST": np.log10(1.0 / p2st),
        }
    )

    return config


def get_cov_from_co2(co2: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """
    Calculate CO2-based covariate.

    Args:
        co2: CO2 levels as a numpy array.
        config: Configuration dictionary.

    Returns:
        Normalized covariate values as a numpy array.
    """
    pre_industrial = config["pre_industrial_co2"]
    current = config["current_co2"]
    cov = np.log(co2 / pre_industrial) / np.log(current / pre_industrial)
    return cov - 0.5


def get_cov_from_year(year: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """
    Calculate year-based covariate.

    Args:
        year: Years as a numpy array.
        config: Configuration dictionary.

    Returns:
        Normalized covariate values as a numpy array.
    """
    base = config["base_year"]
    current = config["current_year"]
    cov = (year - base) / (current - base)
    return cov - 0.5


def get_map_extent(config: Dict[str, Any]) -> List[float]:
    """
    Get map extent for region.

    Args:
        config: Configuration dictionary.

    Returns:
        Map extent [minlon, maxlon, minlat, maxlat].
    """
    extents = config["map_extents"]
    sv = config["region"]

    if sv == "V12" and config.get("nopt", 0):
        return extents["V12_MT"]
    return extents.get(sv, extents["CONUS"])


def _format_year_range(config: Dict[str, Any]) -> str:
    """
    Format year range for directory naming.

    Args:
        config: Configuration dictionary.

    Returns:
        Formatted year range string.
    """
    year_range = config["ams_year_range"]
    if isinstance(year_range, str):
        year_range = [int(x) for x in year_range.split(",")]
    elif isinstance(year_range[0], str):
        year_range = [int(x) for x in year_range]

    start, end = year_range
    if start > 0 and end > 0:
        return f"_{start}-{end}"
    if start > 0:
        return f"_ge{start}"
    if end > 0:
        return f"_le{end}"
    return ""


def _add_model_specific_parts(parts: List[str], config: Dict[str, Any]) -> None:
    """
    Helper method for model-specific directory naming.

    Args:
        parts: List of parts that form the directory name.
        config: Configuration dictionary.
    """
    ok_used_mam = False

    if config["nmodel"] > 3:
        if config["ncov1_option"] == 0:
            map_option = {1: "_sMAP", 2: "_cMAP"}.get(config["nMAPOption"], "_MAP")
            parts.append(map_option)
        elif config["ncov1_option"] == 1:
            ok_used_mam = True
            suffix = "_sMAM" if config["use_station_mam"] else "_pMAM"
            parts.append(suffix)
        else:
            parts.append("_Elev")

    if config["okUseCovMAM"] and config["use_station_mam"] and not ok_used_mam:
        parts.append("_sMAM")

    if config.get("use_greg_mam", True) and config["okUseCovMAM"]:
        parts.append("_Greg")

    if config["ncov2_option"] >= 0:
        cov2_options = {0: "_CO2", 1: "_GWL", 2: "_year"}
        parts.append(cov2_options.get(config["ncov2_option"], ""))

        if config["ncov2_option"] == 1 and config["smooth_gwl"]:
            parts.append("s")

        if config["ncov2term_option"] == 1:
            parts.append("p")


def find_output_dir(config: Dict[str, Any]) -> str:
    """
    Construct output directory name based on configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Output directory name.
    """
    parts = [config["region"]]
    parts.append("_Ka4" if config["fit_kappa"] else "_GEV")
    parts.append("o" if config["out_xobs"] else "")

    if config["solve_at_station"]:
        parts.append("_Sta")
        if config["remove_center_station"]:
            parts.append("0")
    else:
        parts.append("_Grd")

    parts.append("_Gw")
    if config["compute_2sample_tests"]:
        parts.append("St")
    if config["use_dist2coast"]:
        parts.append("Dc")
    if config["arithmetic_mean_weights"]:
        parts.append("Am")

    if config["use_weights_minmax_perc"]:
        parts.append(f"_{config['wmin_perc']}-{config['wmax_perc']}")

    parts.append(f"_M{config['nmodel']}")
    if not config["relax_bounds"]:
        parts.append("b")
        if config["bounds_percentile"] == 1:
            parts.append("1")

    _add_model_specific_parts(parts, config)

    year_range = _format_year_range(config)
    if year_range:
        parts.append(year_range)

    return "".join(parts)


def get_param_perc(config: Dict[str, Any]) -> np.ndarray:
    """
    Get parameter percentiles from fitted data.

    Args:
        config: Configuration dictionary.

    Returns:
        Parameter percentiles as a numpy array.
    """
    file_path = Path(config["proposedOutputPath"]).parent / "fitted_param_percentiles.csv"
    param_perc: List[List[float]] = []

    if file_path.is_file():
        df_perc = pd.read_csv(file_path)

        filtered_df = df_perc[
            (df_perc["case"] == find_output_dir(config))
            & (df_perc["dur"] == config["ams_duration"])
        ]

        if len(filtered_df) >= 3:
            for k in range(len(filtered_df)):
                sub_perc = filtered_df[filtered_df["param"] == f"par{k}"]
                if not sub_perc.empty:
                    if config["bounds_percentile"] == 1:
                        param_perc.append(
                            [
                                sub_perc["p01"].values[0],
                                sub_perc["p50"].values[0],
                                sub_perc["p99"].values[0],
                            ]
                        )
                    elif config["bounds_percentile"] == 0.5:
                        param_perc.append(
                            [
                                sub_perc["p005"].values[0],
                                sub_perc["p50"].values[0],
                                sub_perc["p995"].values[0],
                            ]
                        )
                    else:
                        param_perc = []
                else:
                    logger.warning("Percentiles not found for parameter 'par%s'", k)
        else:
            logger.warning("Insufficient filtered data entries")
    else:
        logger.warning("File not found: %s", file_path)

    return np.array(param_perc)


def backup_existing_file(
    directory_path: Path, file_name: str, extension: str, config: Dict[str, Any]
) -> None:
    """
    Backup existing file with .bak extension.

    Args:
        directory_path: Directory containing the file.
        file_name: Name of file to backup.
        extension: File extension.
        config: Configuration dictionary.
    """
    original_file = directory_path / file_name
    if original_file.exists():
        backup_file_name = file_name.replace(extension, f".bak{extension}")
        backup_file_path = directory_path / backup_file_name

        while backup_file_path.exists():
            backup_file_name = backup_file_name.replace(extension, f".bak{extension}")
            backup_file_path = directory_path / backup_file_name

        try:
            original_file.rename(backup_file_path)
        except OSError as e:
            logger.exception("Failed to backup existing file.")
            raise RuntimeError("File backup failed") from e


def get_log_filename(csv_input_file_name: str, sdur: str, config: Dict[str, Any]) -> str:
    """
    Generate log filename from input filename and duration.

    Args:
        csv_input_file_name: Input CSV filename.
        sdur: Duration string.
        config: Configuration dictionary.

    Returns:
        Generated log filename.
    """
    return csv_input_file_name.replace(".csv", f"_log_{sdur}.csv")


def convert_string_to_list(s: str, config: Dict[str, Any]) -> List[str]:
    """
    Convert string to list of values.

    Args:
        s: String to convert.
        config: Configuration dictionary.

    Returns:
        Converted values as a list of strings.
    """
    return [x.strip() for x in s.split(",")]
