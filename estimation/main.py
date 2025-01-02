import logging
import math
import time
from pathlib import Path
from typing import List, Tuple, Union

# import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

# sys.path.append("/home/rama.mantripragada/a14v13")
from preprocess.covariates_processing import (
    get_cov_from_elev,
    get_cov_from_mam,
    get_cov_from_map,
)

from .functions import (
    fit_distribution,
    get_4pars_from_xpars,
    get_cor_aic_bic,
    get_expected_return_period_from_nams,
    get_fit_distribution_errors,
    get_fit_distribution_errors_station,
    get_precip_from_return_period,
    get_precip_from_x,
    get_return_period_from_precip,
    get_x_from_precip,
)

logger = logging.getLogger(__name__)


def parse_options(config: dict) -> Tuple[str, List[float], List[float]]:
    """
    Extract necessary options from config dictionary and validate them.

    Parameters
    ----------
    config : dict
        Dictionary containing the relevant configuration keys.
        Required keys: 'ams_duration', 'cov2_out', 'return_period_out'

    Returns
    -------
    s_dur : str
        Duration string (e.g. '24h', '60m').
    cov2_out : list of float
        Covariate-2 output values (could be a single value or a list).
    t_out : list of float
        Return periods or similar target values (could be a single value or a list).
    """
    try:
        s_dur = config["ams_duration"]
        cov2_out = config["cov2_out"]
        t_out = config["return_period_out"]
    except KeyError as exc:
        logger.error(
            "config must contain 'ams_duration', 'cov2_out', and 'return_period_out'."
        )
        raise ValueError("Missing required config keys.") from exc

    # If these are not lists already, convert them for consistency
    if not isinstance(cov2_out, list):
        cov2_out = [cov2_out]
    if not isinstance(t_out, list):
        t_out = [t_out]

    return s_dur, cov2_out, t_out


def get_grid_center_info(
    k: int, df_grid: pd.DataFrame
) -> Tuple[float, float, float, float, float]:
    """
    Retrieve MAM, LAT, LON, MAP, and elevation for the given grid index k.

    Returns
    -------
    mam0 : float
    lat : float
    lon : float
    map0 : float
    elev : float
    """
    try:
        mam0 = df_grid.at[k, "gridMAM"]
        lat = df_grid.at[k, "LAT"]
        lon = df_grid.at[k, "LON"]
        map0 = df_grid.at[k, "prismMAP"]
        elev = df_grid.at[k, "elev_DEM"]
    except KeyError as exc:
        logger.error(
            "df_grid missing one or more required columns: "
            "'gridMAM', 'LAT', 'LON', 'prismMAP', 'elev_DEM'."
        )
        raise ValueError("df_grid missing required columns.") from exc

    return mam0, lat, lon, map0, elev


def update_mam_for_station_if_needed(
    k: int,
    df_grid: pd.DataFrame,
    df_meta: pd.DataFrame,
    config: dict,
    mam0: float,
    ok_mc: bool,
) -> Union[Tuple[float, int], None]:
    """
    If config['solve_at_station'] is True, update MAM from df_meta if station ID exists.

    Parameters
    ----------
    k : int
        Grid index
    df_grid : pd.DataFrame
    df_meta : pd.DataFrame
    config : dict
        Should contain 'solve_at_station' key (bool).
    mam0 : float
        Existing MAM value
    ok_mc : bool
        True if part of Monte Carlo simulation
    """
    solve_at_station = config.get("solve_at_station", False)
    if not solve_at_station:
        return (mam0, None)

    try:
        id_num0 = df_grid.at[k, "id_num"]
    except KeyError as exc:
        logger.error("df_grid must contain 'id_num' when solve_at_station is True.")
        raise ValueError("df_grid missing 'id_num' column.") from exc

    # If station is in df_meta, update MAM
    if id_num0 in df_meta.index:
        mam0 = df_meta.loc[id_num0, "usedMAM"]
    else:
        # If station not in df_meta, skip if no AMS
        if ok_mc:
            return None  # skip entirely
        return None

    return (mam0, id_num0)


def build_initial_output_string(
    k: int, lat: float, lon: float, map0: float, mam0: float, elev: float
) -> str:
    """
    Build an initial CSV-like output string with basic region info.
    """
    return f"{k},{lat},{lon},{map0},{mam0},{elev}"


def prepare_neighbor_weights(k: int, df_neighbor_weights: pd.DataFrame) -> pd.DataFrame:
    """
    Subset neighbor weights for grid index k.
    Sets 'id_num' as the index if it exists.
    """
    try:
        sub_neighbor_weights = df_neighbor_weights[
            df_neighbor_weights["k"] == int(k)
        ].copy()
    except KeyError as exc:
        logger.error("df_neighbor_weights must contain 'k' column.")
        raise ValueError("df_neighbor_weights missing 'k' column.") from exc

    if "id_num" in sub_neighbor_weights.columns:
        sub_neighbor_weights.set_index("id_num", inplace=True)

    return sub_neighbor_weights


def merge_meta_with_neighbors(
    df_meta: pd.DataFrame, sub_neighbor_weights: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge station metadata with the neighbor weights to form sub_meta.
    Sorts by distance 'd'.
    """
    df_meta_reset = df_meta.reset_index()  # ensure 'id_num' is a column
    try:
        sub_meta = df_meta_reset.merge(
            sub_neighbor_weights[["k", "d", "w"]], how="inner", on="id_num"
        ).copy()
    except KeyError as exc:
        logger.error(
            "Columns 'id_num', 'k', 'd', 'w' must exist in df_meta "
            "and neighbor weights for merging."
        )
        raise ValueError(
            "Missing columns required for merging neighbor weights."
        ) from exc

    sub_meta.sort_values(by="d", inplace=True)
    return sub_meta


def compute_cov1_if_needed(config: dict, mam0: float, map0: float, elev: float) -> float:
    """
    Compute cov1_out based on config settings. If not in use, returns 0.0.
    """
    ok_use_cov1 = config.get("okUseCov1", False)
    n_cov1_option = config.get("ncov1_option", 0)
    n_map_option = config.get("nMAPOption", 0)

    if not ok_use_cov1:
        return 0.0

    if n_cov1_option == 0:
        return get_cov_from_map(map0, n_map_option)
    if n_cov1_option == 1:
        return get_cov_from_mam(mam0)
    if n_cov1_option == 2:
        return get_cov_from_elev(elev)
    logger.warning("Unknown ncov1_option. Setting cov1_out to 0.0 by default.")
    return 0.0


def add_station_info_to_output(
    s_out: str,
    id_num0: int,
    mam0: float,
    df_meta: pd.DataFrame,
    df_grid: pd.DataFrame,
    k: int,
) -> Tuple[str, int]:
    """
    Add station metadata to the output string if a center station is used.
    Returns updated s_out and the station's nAMS value.
    """
    s_out += f",{id_num0}"
    n_ams0 = 0
    if id_num0 in df_meta.index:
        hdsc_str = df_meta.loc[id_num0, "HDSC"] if "HDSC" in df_meta.columns else ""
        s_out += f",{hdsc_str}"
        n_ams0 = df_meta.loc[id_num0, "nAMS"]
        s_out += f",{n_ams0},{mam0}"
    else:
        hdsc_str = df_grid.at[k, "HDSC"] if "HDSC" in df_grid.columns else "nan"
        s_out += f",{hdsc_str},0,{mam0}"

    return s_out, n_ams0


def remove_center_station_if_needed(
    sub_meta: pd.DataFrame, config: dict, id_num0: int
) -> pd.DataFrame:
    """
    Remove the center station from sub_meta if config['remove_center_station'] is True.
    """
    ok_remove_center_station = config.get("remove_center_station", False)
    if ok_remove_center_station and (id_num0 in sub_meta["id_num"].values):
        sub_meta = sub_meta[sub_meta["id_num"] != id_num0]
    return sub_meta


def prepare_sub_ams(sub_meta: pd.DataFrame, df_ams: pd.DataFrame) -> pd.DataFrame:
    """
    Merge sub_meta with df_ams to get the AMS records for neighbor stations only.
    Remove zero-weight rows.
    """
    df_ams_reset = df_ams.reset_index()  # ensure 'id_num' is a column
    try:
        sub_ams = df_ams_reset.merge(sub_meta[["id_num", "w"]], how="inner", on="id_num")
    except KeyError as exc:
        logger.error("df_ams and sub_meta must contain 'id_num' and 'w' columns.")
        raise ValueError("Missing columns required for merging AMS data.") from exc

    # Remove rows with zero weight

    return sub_ams[sub_ams["w"] > 0]


def append_basic_counts_to_output(
    s_out: str, sub_meta: pd.DataFrame, sub_ams: pd.DataFrame
) -> Tuple[str, float]:
    """
    Append nSta, nAMS, and sum of weights to the output string.
    Returns updated s_out and the sum of weights.
    """
    n_sta = len(sub_meta.index)
    n_ams = len(sub_ams.index)
    sum_w = sub_ams["w"].sum()
    s_out += f",{n_sta},{n_ams},{sum_w}"
    return s_out, sum_w


def reevaluate_cov1_if_mam(
    config: dict,
    sub_meta: pd.DataFrame,
    sub_ams: pd.DataFrame,
    mam0: float,
    cov1_out: float,
) -> float:
    """
    If ncov1_option == 1 (MAM), recalculate cov1 for sub_meta & sub_ams,
    fill NaNs, and recalc cov1_out if needed.
    """
    ok_use_cov1 = config.get("okUseCov1", False)
    n_cov1_option = config.get("ncov1_option", 0)

    if ok_use_cov1 and n_cov1_option == 1:
        sub_meta["cov1"] = get_cov_from_mam(sub_meta["usedMAM"])
        cov1_mean = sub_meta["cov1"].mean()
        sub_meta["cov1"].fillna(cov1_mean, inplace=True)

        sub_ams["cov1"] = get_cov_from_mam(sub_ams["usedMAM"])
        sub_ams["cov1"].fillna(cov1_mean, inplace=True)

        out_val = get_cov_from_mam(mam0)
        if math.isnan(out_val):
            out_val = cov1_mean
        return out_val

    return cov1_out


def do_distribution_fitting(
    sub_ams: pd.DataFrame, config: dict, s_dur: str, sum_w: float
) -> Tuple[np.ndarray, float, int]:
    """
    Fit the distribution if sum_w > 0.
    Returns (x_0, m_mle, n_iter_minimize).
    If no neighbors, returns ([nan, nan, nan], nan, 0).
    """
    if sum_w <= 0:
        return np.full(3, np.nan), np.nan, 0

    try:
        fit_kappa = config["fit_kappa"]
        n_model = config["nmodel"]
        ok_use_cov2 = config["okUseCov2"]
        n_cov2_term_option = config["ncov2term_option"]
        ok_relax_bnds = config["relax_bounds"]
        param_perc = config["param_perc"]
    except KeyError as exc:
        logger.error("config missing required distribution fitting keys.")
        raise ValueError("config missing required distribution fitting fields.") from exc

    try:
        x_0, m_mle, n_iter_minimize = fit_distribution(
            sub_ams["precip"].values,
            fit_kappa,
            n_model,
            ok_use_cov2,
            n_cov2_term_option,
            sub_ams["usedMAM"].values,
            sub_ams["prismMAP"].values,
            sub_ams.get("cov1", 0.0).values,
            sub_ams.get("cov2", 0.0).values,
            sub_ams["w"].values,
            0.0,
            0.0,
            False,
            ok_relax_bnds,
            s_dur,
            param_perc,
        )
    except Exception as exc:
        logger.error("fitDistribution failed.")
        raise RuntimeError("fitDistribution encountered an error.") from exc

    return x_0, m_mle, n_iter_minimize


def do_post_processing(
    x_0: np.ndarray,
    m_mle: float,
    n_iter_minimize: int,
    sum_w: float,
    s_out: str,
    sub_ams: pd.DataFrame,
    sub_meta: pd.DataFrame,
    mam0: float,
    map0: float,
    cov1_out: float,
    cov2_out: List[float],
    config: dict,
    s_point: str,
    s_dur: str,
) -> str:
    """
    Perform all post-fit calculations (expanding parameters, calculating errors, etc.)
    and update the output string. Returns the updated s_out.
    """
    # 1) Append iteration info & parameter estimates
    s_out += f",{n_iter_minimize}"
    s_out += "," + ",".join(str(xx) for xx in x_0)

    # 2) If no neighbors, fill with placeholders
    if sum_w <= 0:
        s_out += ",nan,nan,nan,nan,nan,nan,nan,nan,nan"
        return s_out

    # 3) Post-processing
    try:
        fit_kappa = config["fit_kappa"]
        n_model = config["nmodel"]
        ok_use_cov2 = config["okUseCov2"]
        n_cov2_term_option = config["ncov2term_option"]

        (sub_ams["loc"], sub_ams["scl"], sub_ams["shape"], sub_ams["shape2"]) = (
            get_4pars_from_xpars(
                x_0,
                fit_kappa,
                n_model,
                ok_use_cov2,
                n_cov2_term_option,
                sub_ams["usedMAM"].values,
                sub_ams["prismMAP"].values,
                sub_ams.get("cov1", 0.0).values,
                sub_ams.get("cov2", 0.0).values,
            )
        )

        # Standardized variable x
        sub_ams["x"] = get_x_from_precip(
            sub_ams["precip"].values, sub_ams["loc"].values, sub_ams["scl"].values
        )

        # Return period T
        sub_ams["Tt"] = get_return_period_from_precip(
            sub_ams["precip"].values,
            sub_ams["loc"].values,
            sub_ams["scl"].values,
            sub_ams["shape"].values,
            sub_ams["shape2"].values,
            fit_kappa,
        )

        # Sort by standardized variable x
        sub_ams.sort_values(by="x", inplace=True)

        # Expected return periods
        n_ams = len(sub_ams.index)
        sub_ams["Te"] = get_expected_return_period_from_nams(n_ams)

        # Theoretical standardized precip
        sub_ams["xt"] = get_precip_from_return_period(
            sub_ams["Te"].values,
            0.0,
            1.0,
            sub_ams["shape"].values,
            sub_ams["shape2"].values,
            fit_kappa,
        )

        # Theoretical precip in inches
        sub_ams["pt"] = get_precip_from_return_period(
            sub_ams["Te"].values,
            sub_ams["loc"].values,
            sub_ams["scl"].values,
            sub_ams["shape"].values,
            sub_ams["shape2"].values,
            fit_kappa,
        )

        # Fitting errors & Info Criteria
        m_mle_w = m_mle / sum_w
        aicc, bic = get_cor_aic_bic(m_mle, len(x_0), sum_w)
        s_out += f",{m_mle_w},{aicc},{bic}"

        # Distribution parameters at region center
        loc0, scl0, shape0, shape20 = get_4pars_from_xpars(
            x_0,
            fit_kappa,
            n_model,
            ok_use_cov2,
            n_cov2_term_option,
            mam0,
            map0,
            cov1_out,
            cov2_out,
        )

        # If scl0 is an array (depending on how cov2 is used), reduce to mean
        if ok_use_cov2 and hasattr(scl0, "__len__"):
            scl0 = np.mean(scl0)

        # Compute fit errors (regional)
        me, mae, rmse = get_fit_distribution_errors(
            sub_ams["x"].values,
            sub_ams["xt"].values,
            sub_ams["Te"].values,
            sub_ams["w"].values,
            scl0,
            shape0,
            config.get("errors_in_inches", False),
            config.get("save_plots", False),
            f"{s_point}, dur={s_dur}",
            Path(config.get("proposedOutputPath", "."), s_dur, "x_vs_T_plots"),
        )
        s_out += f",{me},{mae},{rmse}"

        # Compute fit errors by station
        (me_sta, mae_sta, rmse_sta, me_a, mae_a, rmse_a) = (
            get_fit_distribution_errors_station(
                sub_ams.copy(),
                sub_meta["id_num"].unique(),
                fit_kappa,
                scl0,
                shape0,
                config.get("errors_in_inches", False),
                config.get("save_plots", False),
                f"{s_point} bySta, dur={s_dur}",
                (n_model == 0),
                Path(config.get("proposedOutputPath", "."), s_dur, "x_vs_T_plots_bySta"),
            )
        )
        s_out += f",{me_sta},{mae_sta},{rmse_sta}"

    except Exception as exc:
        logger.error("Post-processing after fit distribution failed.")
        raise RuntimeError("Error in post-processing distribution fits.") from exc

    return s_out


def append_cov1_if_used(s_out: str, config: dict, cov1_out: float) -> str:
    """
    Append cov1_out to the output string if 'okUseCov1' is True.
    """
    if config.get("okUseCov1", False):
        s_out += f",{cov1_out}"
    return s_out


def estimate_precip_from_distribution(
    x_0: np.ndarray,
    sum_w: float,
    config: dict,
    sub_ams: pd.DataFrame,
    t_out: List[float],
    cov2_out: List[float],
    mam0: float,
    map0: float,
    cov1_out: float,
    s_out: str,
) -> str:
    """
    Use the fitted distribution parameters (x_0) to estimate precipitation
    for specified return periods (t_out) and covariate values (cov2_out).
    """
    ok_mc = config.get("okMC", False)
    ok_out_xobs = config.get("out_xobs", False)
    ok_use_cov2 = config.get("okUseCov2", False)
    fit_kappa = config.get("fit_kappa", False)
    n_model = config.get("nmodel", 0)
    n_cov2_term_option = config.get("ncov2term_option", 0)

    # If we want to interpolate from observed data at certain return_period_out
    if (not ok_mc) and ok_out_xobs:
        try:
            sub_ams.sort_values(by="Te", inplace=True)
            x_out = np.interp(t_out, sub_ams["Te"].values, sub_ams["x"].values)
        except Exception:
            logger.warning("Interpolation failed, setting x_out to nan.")
            x_out = np.full(len(t_out), np.nan)
    else:
        # Use placeholder
        x_out = np.full(len(t_out), np.nan)

    # For each cov2 in cov2_out
    if ok_use_cov2:
        for c2_val in cov2_out:
            if sum_w > 0:
                loc_out, scl_out, shp, shp2 = get_4pars_from_xpars(
                    x_0,
                    fit_kappa,
                    n_model,
                    True,
                    n_cov2_term_option,
                    mam0,
                    map0,
                    cov1_out,
                    c2_val,
                )
            else:
                loc_out, scl_out, shp, shp2 = (np.nan, np.nan, np.nan, np.nan)

            s_out += f",{loc_out},{scl_out}"

            for j, tt in enumerate(t_out):
                if sum_w > 0:
                    if ok_out_xobs:
                        rk = get_precip_from_x(x_out[j], loc_out, scl_out)
                    else:
                        rk = get_precip_from_return_period(
                            tt, loc_out, scl_out, shp, shp2, fit_kappa
                        )
                        x_out[j] = get_x_from_precip(rk, loc_out, scl_out)
                else:
                    rk = np.nan
                s_out += f",{rk}"
    else:
        # Single cov2_out scenario
        c2_val = cov2_out[0]
        if sum_w > 0:
            loc_out, scl_out, shp, shp2 = get_4pars_from_xpars(
                x_0,
                fit_kappa,
                n_model,
                False,
                n_cov2_term_option,
                mam0,
                map0,
                cov1_out,
                c2_val,
            )
        else:
            loc_out, scl_out, shp, shp2 = (np.nan, np.nan, np.nan, np.nan)

        s_out += f",{loc_out},{scl_out}"
        for j, tt in enumerate(t_out):
            if sum_w > 0:
                if ok_out_xobs:
                    rk = get_precip_from_x(x_out[j], loc_out, scl_out)
                else:
                    rk = get_precip_from_return_period(
                        tt, loc_out, scl_out, shp, shp2, fit_kappa
                    )
                    x_out[j] = get_x_from_precip(rk, loc_out, scl_out)
            else:
                rk = np.nan
            s_out += f",{rk}"

    return s_out


def append_a14_if_needed(s_out: str, config: dict, df_grid: pd.DataFrame, k: int) -> str:
    """
    Append A14 precipitation estimates if available and required.
    """
    ok_mc = config.get("okMC", False)
    ok_output_a14 = config.get("output_a14_precip", False)

    if (not ok_mc) and ok_output_a14:
        if "A14p_2.54y" in df_grid.columns:
            a14_2_54y = df_grid.at[k, "A14p_2.54y"]
            a14_25y = df_grid.at[k, "A14p_25y"]
            a14_100y = df_grid.at[k, "A14p_100y"]
            s_out += f",{a14_2_54y},{a14_25y},{a14_100y}"
        else:
            s_out += ",nan,nan,nan"
    return s_out


def fit_one_region(
    k: int,
    df_grid: pd.DataFrame,
    df_meta: pd.DataFrame,
    df_ams: pd.DataFrame,
    df_neighbor_weights: pd.DataFrame,
    config: dict,
    ok_mc: bool = False,
) -> Union[str, Tuple[str, pd.DataFrame]]:
    """
    Fit a distribution for a given region/grid point based on neighbor stations.

    Parameters
    ----------
    k : int
        Index (or grid point index) to process from df_grid.
    df_grid : pd.DataFrame
        DataFrame containing grid information (MAM, MAP, elevation, etc.).
    df_meta : pd.DataFrame
        DataFrame containing station metadata (MAM, station IDs, etc.).
    df_ams : pd.DataFrame
        DataFrame containing the annual maximum series for each station.
    df_neighbor_weights : pd.DataFrame
        DataFrame containing neighbor weights (and distances) for each grid point.
    config : dict
        Dictionary containing various option flags and parameters for the fitting.
        (e.g., 'ams_duration', 'cov2_out', 'return_period_out', 'solve_at_station', etc.)
    ok_mc : bool, optional
        Flag to indicate if this call is part of a Monte Carlo simulation.

    Returns
    -------
    Union[str, Tuple[str, pd.DataFrame]]
        If `ok_mc` is True, returns a single string output (e.g. for batch runs).
        Otherwise, returns a tuple of (output_string, sub_meta DataFrame).
    """
    start_time = time.time()

    # Store ok_mc in config so helpers can see it if needed
    config["okMC"] = ok_mc

    # 1) Parse options from config
    s_dur, cov2_out, t_out = parse_options(config)

    # 2) Extract info from df_grid
    mam0, lat, lon, map0, elev = get_grid_center_info(k, df_grid)
    s_point = f"Point k={k}"

    # 3) If station-based solve, possibly update mam0
    mam_station_update = update_mam_for_station_if_needed(
        k, df_grid, df_meta, config, mam0, ok_mc
    )
    if mam_station_update is None:
        # Means station not in df_meta or no AMS, so skip
        return "" if ok_mc else ("", [])
    mam0, id_num0 = mam_station_update

    # 4) Build initial output string
    s_out = build_initial_output_string(k, lat, lon, map0, mam0, elev)

    # 5) Prepare neighbor weights and merge with df_meta
    sub_neighbor_weights = prepare_neighbor_weights(k, df_neighbor_weights)
    sub_meta = merge_meta_with_neighbors(df_meta, sub_neighbor_weights)

    # 6) Compute cov1 (if used)
    cov1_out = compute_cov1_if_needed(config, mam0, map0, elev)

    # 7) If station-based solve, add station info to output
    n_ams0 = 0
    if id_num0 is not None:
        s_out, n_ams0 = add_station_info_to_output(
            s_out, id_num0, mam0, df_meta, df_grid, k
        )
        sub_meta = remove_center_station_if_needed(sub_meta, config, id_num0)

        # If center station has no AMS, skip
        if n_ams0 == 0:
            return "" if ok_mc else ("", [])

    # 8) Prepare sub_ams
    sub_ams = prepare_sub_ams(sub_meta, df_ams)

    # 9) Append basic counts (nSta, nAMS, sum_w) to output
    s_out, sum_w = append_basic_counts_to_output(s_out, sub_meta, sub_ams)

    # 10) Re-evaluate cov1 if MAM-based
    cov1_out = reevaluate_cov1_if_mam(config, sub_meta, sub_ams, mam0, cov1_out)

    # 11) Fit distribution (if sum_w > 0)
    x_0, m_mle, n_iter_minimize = do_distribution_fitting(sub_ams, config, s_dur, sum_w)

    # 12) Post-processing
    s_out = do_post_processing(
        x_0,
        m_mle,
        n_iter_minimize,
        sum_w,
        s_out,
        sub_ams,
        sub_meta,
        mam0,
        map0,
        cov1_out,
        cov2_out,
        config,
        s_point,
        s_dur,
    )

    # 13) Append cov1 if used
    s_out = append_cov1_if_used(s_out, config, cov1_out)

    # 14) Estimate precipitation from distribution (return periods, etc.)
    s_out = estimate_precip_from_distribution(
        x_0, sum_w, config, sub_ams, t_out, cov2_out, mam0, map0, cov1_out, s_out
    )

    # 15) Append A14 precipitation if needed
    s_out = append_a14_if_needed(s_out, config, df_grid, k)
    s_out = s_out + "," + str(time.time() - start_time)

    # 16) Return results
    if ok_mc:
        return s_out + "\n"
    return s_out, sub_meta


def fit_regions_parallel(
    df_grid: pd.DataFrame,
    df_meta: pd.DataFrame,
    df_ams: pd.DataFrame,
    df_neighbor_weights: pd.DataFrame,
    config: dict,
    out_fit_file: str,
    out_weight_file: str,
    ok_mc: bool,
    n_jobs: int = -1,
) -> None:
    """Execute regional distribution fitting in parallel using joblib.

    Args:
        df_grid: DataFrame with grid information
        df_meta: DataFrame with station metadata
        df_ams: DataFrame with annual maximum series
        df_neighbor_weights: DataFrame with neighbor weights
        config: Configuration dictionary
        ok_mc: Monte Carlo True or False
        n_jobs: Number of parallel jobs (-1 for all cores)

    Returns:
        Tuple containing:
            - List of output strings from each region
            - Combined DataFrame of sub_meta results
    """
    try:
        # Get list of grid points to process
        grid_points = range(len(df_grid))
        total_points = len(grid_points)

        # Configure progress bar
        results = Parallel(n_jobs=n_jobs)(
            delayed(fit_one_region)(
                k, df_grid, df_meta, df_ams, df_neighbor_weights, config, ok_mc
            )
            for k in tqdm(
                grid_points,
                total=total_points,
                desc="Processing regions",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            )
        )

        # Separate outputs and sub_meta DataFrames
        sub_metas = []
        for s_out, sub_meta in results:
            if s_out:
                with open(out_fit_file, "a") as f:
                    f.write(s_out + "\n")
                if not sub_meta.empty:
                    sub_metas.append(sub_meta)

        # Combine all sub_meta DataFrames
        combined_sub_meta = pd.concat(sub_metas, ignore_index=True)
        combined_sub_meta.to_csv(Path(out_weight_file), index=False)

    except Exception as e:
        logger.error(f"Error in parallel processing: {str(e)}")
        raise
