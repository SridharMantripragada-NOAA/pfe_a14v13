import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import genextreme, kappa4

logger = logging.getLogger(__name__)


def extract_daily_or_subdaily_weights(config: dict, duration: str) -> pd.DataFrame:
    """
    Extract neighbor weights from CSV files based on duration.

    Loads pre-computed neighbor weights from either subdaily or daily CSV files depending
    on the specified duration. The weights are used for precipition frequency analysis.

    Args:
        config (dict): Configuration dictionary containing output path information.
            Must have key 'proposedOutputPath' with valid path string.
        duration (str): Time duration for analysis. Must be one of:
            '60m', '06h' for subdaily analysis
            '24h', '04d', '10d', '60d' for daily analysis

    Returns:
        pd.DataFrame: DataFrame containing neighbor weights indexed by station ID.

    Raises:
        ValueError: If duration is not one of the supported values.
        FileNotFoundError: If weights file does not exist at specified path.
        pd.errors.EmptyDataError: If weights file is empty.
        Exception: For other errors during file reading or processing.
    """
    try:
        # Determine file path based on duration
        if duration in ["60m", "06h"]:
            file_path = Path(config["proposedOutputPath"], "NeighborWeights_subdaily.csv")
        elif duration in ["24h", "04d", "10d", "60d"]:
            file_path = Path(config["proposedOutputPath"], "NeighborWeights_daily.csv")
        else:
            supported_durations = ["60m", "06h", "24h", "04d", "10d", "60d"]
            raise ValueError(
                f"Duration '{duration}' not supported. "
                f"Supported durations are: {supported_durations}"
            )

        if not file_path.exists():
            raise FileNotFoundError(f"Weights file not found: {file_path}")

        # Read and process weights file
        df = pd.read_csv(file_path)

        logger.info(f"Successfully loaded weights from: {file_path}")
        return df

    except FileNotFoundError as e:
        logger.error(f"Weights file not found: {e}")
        raise

    except pd.errors.EmptyDataError:
        logger.error(f"Weights file is empty: {file_path}")
        raise

    except Exception as e:
        logger.error(f"Error extracting weights: {str(e)}")
        raise


def solve_points_one_duration_header(config: Dict[str, Any]) -> str:
    """
    Generate the CSV header line for solving points at one duration.

    This function constructs the header line for a CSV file used in the
    `solve_points_one_duration` process. It dynamically adjusts the header
    based on the configuration options provided.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration parameters containing options and settings.

    Returns
    -------
    str
        A string representing the header line for the CSV file.
    """
    # Initialize header components
    header_items = ["k", "Lat", "Lon", "prismMAP", "gridMAM", "Elev_m"]

    if config.get("solve_at_station", [False]):
        # Add center station parameters
        header_items.extend(["ID0", "HDSC0", "nAMS0", "MAM0"])

    # Add fitting parameters
    header_items.extend(["nStations", "nAMS", "sum_w", "nIterMinimize"])

    # Get initial parameters for the distribution fitting
    x0, _ = get_initial_and_bounds_params(
        fit_kappa=config["fit_kappa"],
        n_model=config["nmodel"],
        ok_use_cov2=config["okUseCov2"],
        mam_mean=1.0,
        map_mean=1.0,
        shape=0.0,
        shape2=0.0,
        ok_fix_shape=False,
        ok_relax_bounds=config["relax_bounds"],
        duration=config["ams_duration"],
    )

    # Add parameter labels
    for idx in range(len(x0)):
        header_items.append(f"par{idx}")

    # Add error indicators
    header_items.extend(["mMLE_w", "AICc", "BIC"])

    # Add error metrics
    if config["errors_in_inches"]:
        header_items.extend(
            ["ME_in", "MAE_in", "RMSE_in", "MEbySta_in", "MAEbySta_in", "RMSEbySta_in"]
        )
    else:
        header_items.extend(["ME", "MAE", "RMSE", "MEbySta", "MAEbySta", "RMSEbySta"])

    # Add covariate if used
    if config["okUseCov1"]:
        header_items.append("cov1")

    return_period_out = config["return_period_out"]
    cov2_out = config["cov2_out"]

    # Add location and scale parameters and precipitation labels
    if config["okUseCov2"]:
        for idx, cov2_value in enumerate(cov2_out):
            header_items.append(f"loc_{cov2_value}")
            header_items.append(f"scl_{cov2_value}")
            for k in range(len(return_period_out)):
                label = get_precip_label(config, k, idx)
                header_items.append(label)
    else:
        header_items.extend(["loc", "scl"])
        for k in range(len(return_period_out)):
            label = get_precip_label(config, k, 0)
            header_items.append(label)

    # Add A14 precipitation estimates if required
    if config["output_a14_precip"]:
        header_items.extend(["A14P_2y", "A14P_25y", "A14P_100y"])

    # Add elapsed time
    header_items.append("elapsedTime(sec)")

    # Join all header items into a single string
    return ",".join(header_items) + "\n"


def get_precip_label(config: Dict[str, Any], k: int, kcov2: int, ncov2: int = 2) -> str:
    """
    Generate precipitation label based on configuration.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration parameters containing options and settings.
    k : int
        Index of the return period in `return_period_out`.
    kcov2 : int
        Index of the covariate 2 value in `cov2_out`.
    ncov2 : int, optional
        Number of covariate 2 terms (default is 2).

    Returns
    -------
    str
        A formatted precipitation label string.

    Notes
    -----
    - The label is constructed based on the return period `return_period_out`
      and covariate 2 values.
    - If `return_period_out` is an integer (within a tolerance), it is formatted
      without decimals.
    - If `okUseCov2` is True and `ncov2` equals 2, the covariate 2 value is appended to
      the label.
    """
    return_period_out_value = config["return_period_out"][k]

    # Check if return_period_out_value is effectively an integer
    if abs(return_period_out_value - round(return_period_out_value)) < 0.01:
        ny = int(round(return_period_out_value))  # Year as integer
    else:
        ny = round(return_period_out_value, 2)  # Year as float with 2 decimal places

    # Construct the base label
    label = f"P_{ny}y_in"

    # Append covariate 2 value if conditions are met
    if config["okUseCov2"] and ncov2 == 2:
        cov2_value = config["cov2_out"][kcov2]
        label += f"_{cov2_value}"

    return label


def get_initial_and_bounds_params(
    fit_kappa: bool,
    n_model: int,
    ok_use_cov2: bool,
    mam_mean: float,
    map_mean: float,
    shape: float,
    shape2: float,
    ok_fix_shape: bool,
    ok_relax_bounds: bool,
    duration: str,
) -> Tuple[np.ndarray, Tuple[Tuple[float, float], ...]]:
    """
    Get initial parameters and bounds for the distribution fitting.

    Parameters
    ----------
    fit_kappa : bool
        Flag indicating whether to fit a Kappa-4 distribution
        (i.e., shape2 parameter is included).
    n_model : int
        Model number indicating the parameter model to use.
    ok_use_cov2 : bool
        Flag indicating whether to use covariate 2.
    mam_mean : float
        Mean Annual Maximum (MAM) value at the location.
    map_mean : float
        Mean Annual Precipitation (MAP) value at the location.
    shape : float
        Shape parameter (shape1).
    shape2 : float
        Second shape parameter (shape2), used when `fit_kappa` is True.
    ok_fix_shape : bool
        Flag indicating whether to fix the shape parameters.
    ok_relax_bounds : bool
        Flag indicating whether to relax the parameter bounds.
    duration : str
        Duration string (e.g., '24h' or '60m').

    Returns
    -------
    x0 : np.ndarray
        Initial parameter values for the fitting process.
    bounds : Tuple[Tuple[float, float], ...]
        Bounds for each parameter.

    Raises
    ------
    ValueError
        If an unrecognized `n_model` is provided or `duration` is not in the
        duration list.
    """
    # Median values obtained from fitting GEV distribution to single stations
    # in the A14 Vol12 area
    sdur = ["60m", "06h", "24h", "04d", "10d", "60d"]
    loc_median = [0.3280, 0.7236, 1.0792, 1.6946, 2.2816, 5.4618]
    scl_median = [0.1171, 0.1978, 0.3239, 0.5406, 0.7218, 1.6327]
    shape_median = [-0.2207, -0.1039, -0.0562, -0.0485, -0.0270, 0.0527]
    mam_median = [0.4294, 0.8627, 1.3045, 2.0683, 2.7561, 6.3762]

    # Find index corresponding to the provided duration
    try:
        kd = sdur.index(duration)
    except ValueError:
        raise ValueError(
            f"Duration '{duration}' not recognized. Valid options are {sdur}."
        )

    # Initialize loc0 and scl0 based on the model
    if n_model == 1:
        # loc proportional to MAM and scl proportional to loc
        loc0 = loc_median[kd] / mam_median[kd]
        scl0 = scl_median[kd] / loc_median[kd]
    elif n_model in [2, 21, 22]:
        # loc and scl proportional to MAM
        loc0 = loc_median[kd] / mam_median[kd]
        scl0 = scl_median[kd] / mam_median[kd]
    elif n_model == 3:
        # loc and scl proportional to MAP
        loc0 = (loc_median[kd] / mam_median[kd]) * (mam_mean / map_mean)
        scl0 = (scl_median[kd] / mam_median[kd]) * (mam_mean / map_mean)
    elif n_model in [0, 4]:
        # Default option with uniform parameters
        loc0 = (loc_median[kd] / mam_median[kd]) * mam_mean
        scl0 = (scl_median[kd] / mam_median[kd]) * mam_mean
    else:
        raise ValueError(f"Model number '{n_model}' not recognized.")

    # Set boundaries for loc and scl parameters
    loc_bounds = (loc0 * 0.01, loc0 * 100)
    scl_bounds = (scl0 * 0.01, scl0 * 100)

    # Initialize shape parameters
    shape0 = shape_median[kd]
    shape_bounds = (-0.499, 0.499)
    shape2_0 = 0.0
    shape2_bounds = (-0.499, 0.499)

    if ok_fix_shape:
        # Use provided shape parameters
        shape0 = shape
        shape2_0 = shape2

    if ok_relax_bounds:
        # Relax boundaries for loc, scl, and shape parameters
        loc_bounds = (0.001, 1000)
        scl_bounds = (0.001, 1000)
        shape_bounds = (-0.499, 0.499)
        shape2_bounds = (-0.499, 0.499)

    # Assemble initial parameters array
    x0 = [loc0, scl0, shape0]
    if fit_kappa:
        x0.append(shape2_0)

    # Add correction terms for certain models
    if n_model in [21, 22, 4]:
        x0.extend([0.0, 0.0])  # Starting values for correction terms

    # Add temporal covariate terms if used
    if ok_use_cov2:
        x0.extend([0.0, 0.0])  # Starting values for temporal covariate terms

    x0 = np.array(x0)

    # Establish parameter bounds for optimization
    if ok_relax_bounds:
        # Wide boundaries
        bounds = [(-5.0, 5.0) for _ in x0]
    else:
        # Tight boundaries
        bounds = [(-1.0, 1.0) for _ in x0]

    bounds[0] = loc_bounds  # loc parameter bounds
    bounds[1] = scl_bounds  # scl parameter bounds
    bounds[2] = shape_bounds  # shape parameter bounds

    if fit_kappa:
        bounds[3] = shape2_bounds  # shape2 parameter bounds

    bounds = tuple(bounds)

    return x0, bounds


def get_initial_and_bounds_params_perc(
    param_perc: Any,
) -> Tuple[np.ndarray, Tuple[Tuple[float, float], ...]]:
    """
    Extract initial parameters and parameter bounds from given percentiles.

    This function takes a set of parameters, each defined by three percentiles
    (e.g., 1st, 50th, and 99th percentiles), and returns:
    - The initial parameter values (x0) chosen from the 50th percentile.
    - The parameter bounds derived from the 1st and 99th percentiles.

    Parameters
    ----------
    param_perc : array-like
        An array-like object with shape (N, 3), where each row corresponds
        to a parameter and contains three values: [p1, p50, p99]
        representing the 1st, 50th, and 99th percentiles respectively.

    Returns
    -------
    x0 : np.ndarray
        A 1D array of length N containing the initial parameter values
        taken from the 50th percentiles.
    bnds : tuple of tuples
        A tuple of length N, where each element is a 2-element tuple
        specifying the lower (1st percentile) and upper (99th percentile)
        bounds for the corresponding parameter.

    Raises
    ------
    ValueError
        If `param_perc` does not have shape (N, 3).

    Examples
    --------
    >>> param_perc = [
    ...     [0.1, 0.5, 1.0],   # parameter 1 percentiles
    ...     [-0.2, 0.0, 0.2]   # parameter 2 percentiles
    ... ]
    >>> x0, bnds = get_initial_and_bounds_params_perc(param_perc)
    >>> x0
    array([0.5, 0.0])
    >>> bnds
    ((0.1, 1.0), (-0.2, 0.2))
    """
    arr = np.array(param_perc)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("param_perc must be of shape (N, 3).")

    x0 = arr[:, 1]  # Extract the 50th percentile for each parameter
    # Use the 1st and 99th percentiles as lower and upper bounds
    bnds = tuple((row[0], row[2]) for row in arr)

    return x0, bnds


def fit_distribution(
    precip: np.ndarray,
    fit_kappa: bool,
    n_model: int,
    ok_use_cov2: bool,
    n_cov2_term_option: int,
    mam: np.ndarray,
    map_values: np.ndarray,
    cov1: np.ndarray,
    cov2: np.ndarray,
    weights: np.ndarray,
    shape: float,
    shape2: float,
    ok_fix_shape: bool,
    ok_relax_bounds: bool,
    sdur: str,
    param_perc: np.ndarray,
) -> Tuple[np.ndarray, float, int]:
    """
    Fit the stationary distribution for AMS values.

    Parameters
    ----------
    precip : np.ndarray
        Array of precipitation values.
    fit_kappa : bool
        Flag indicating whether to fit a Kappa-4 distribution.
    n_model : int
        Model number indicating the distribution to use.
    ok_use_cov2 : bool
        Flag indicating whether to use covariate 2.
    n_cov2_term_option : int
        Covariate 2 term option.
    mam : np.ndarray
        Array of Mean Annual Maximum (MAM) values.
    map_values : np.ndarray
        Array of Mean Annual Precipitation (MAP) values.
    cov1 : np.ndarray
        Array of covariate 1 values.
    cov2 : np.ndarray
        Array of covariate 2 values.
    weights : np.ndarray
        Array of weights for each data point.
    shape : float
        Shape parameter (shape1).
    shape2 : float
        Second shape parameter (shape2), used when fit_kappa is True.
    ok_fix_shape : bool
        Flag indicating whether to fix the shape parameters.
    ok_relax_bounds : bool
        Flag indicating whether to relax the parameter bounds.
    sdur : str
        Duration string (e.g., '24h' or '60m').
    param_perc : np.ndarray
        Array of parameter percentiles.

    Returns
    -------
    x0 : np.ndarray
        Optimized parameters.
    m_mle : float
        Final value of the objective function.
    niter_minimize : int
        Number of iterations in the minimization process.

    Raises
    ------
    Exception
        If an error occurs during optimization.
    """
    import numpy as np

    try:
        # Validate input arrays
        if not (len(precip) > 0 and len(mam) > 0 and len(map_values) > 0):
            raise ValueError(
                "Input arrays precip, mam, and map_values must not be empty."
            )
        # mam_mean = np.mean(mam)
        # map_mean = np.mean(map_values)
        # Initial parameters and bounds
        x0, bounds = get_initial_and_bounds_params(
            fit_kappa,
            n_model,
            ok_use_cov2,
            np.mean(mam),
            np.mean(map_values),
            shape,
            shape2,
            ok_fix_shape,
            ok_relax_bounds,
            sdur,
        )
        logger.debug(f"Initial parameters x0: {x0}, bounds: {bounds}")

        # Adjust initial parameters and bounds based on percentiles if provided
        if param_perc.size > 0 and not np.isnan(param_perc[0][0]):
            # Initial parameters and bounds from percentiles
            x0_perc, bounds_perc = get_initial_and_bounds_params_perc(param_perc)
            if len(x0) == len(x0_perc):
                x0 = x0_perc  # Use 50th percentiles as initial values
                logger.debug(f"Adjusted initial parameters x0 from percentiles: {x0}")
                if not ok_relax_bounds:
                    bounds = bounds_perc  # Use 1st and 99th percentiles as bounds
                    logger.debug(f"Adjusted bounds from percentiles: {bounds}")

        # Evaluate the initial objective function value
        m_mle = nopenlik(
            x0,
            precip,
            fit_kappa,
            n_model,
            ok_use_cov2,
            n_cov2_term_option,
            mam,
            map_values,
            cov1,
            cov2,
            weights,
            shape,
            shape2,
            ok_fix_shape,
        )
        logger.debug(f"Initial objective function value m_mle: {m_mle}")

        # Optimization loop
        ok_continue = True
        niter_minimize = 0
        max_iterations = 100  # Maximum number of iterations to prevent infinite loops
        while ok_continue and niter_minimize < max_iterations:
            # Minimize the negative log-likelihood function
            res_ams = minimize(
                nopenlik,
                x0,
                args=(
                    precip,
                    fit_kappa,
                    n_model,
                    ok_use_cov2,
                    n_cov2_term_option,
                    mam,
                    map_values,
                    cov1,
                    cov2,
                    weights,
                    shape,
                    shape2,
                    ok_fix_shape,
                ),
                method="Nelder-Mead",
                options={"return_all": False},
                bounds=bounds,
            )
            m_mle = res_ams.fun
            if ok_fix_shape:
                # Fix the shape parameters
                res_ams.x[2] = shape
                if fit_kappa and len(res_ams.x) > 3:
                    res_ams.x[3] = shape2

            # Check for convergence
            delta_x = res_ams.x - x0
            ok_continue = np.std(delta_x) > 1e-4
            niter_minimize += 1
            logger.debug(
                f"Iteration {niter_minimize}: m_mle={m_mle}, x0={res_ams.x}, "
                f"delta_x std={np.std(delta_x)}"
            )
            x0 = res_ams.x

        if niter_minimize >= max_iterations:
            logger.warning("Maximum number of iterations reached in fit_distribution.")

        return x0, m_mle, niter_minimize

    except Exception as e:
        logger.exception("Error in fit_distribution: %s", e)
        raise


def nopenlik(
    xpars: np.ndarray,
    data: np.ndarray,
    fit_kappa: bool,
    n_model: int,
    ok_use_cov2: bool,
    n_cov2_term_option: int,
    mam: np.ndarray,
    map: np.ndarray,
    cov1: np.ndarray,
    cov2: np.ndarray,
    w: np.ndarray,
    shp: float,
    shp2: float,
    ok_fix_shp: bool,
) -> float:
    """
    Objective function (negative log-likelihood) for fitting distribution
    parameters to data.

    This function computes the negative of the weighted log-likelihood based on the
    given parameters `xpars`, observed data `data`, and a set of configuration flags
    and additional arrays. The distribution can be a GEV (Generalized Extreme Value)
    or a Kappa-4 distribution, depending on the input flags. Spatial weights `w` are
    applied to the data points, and the function returns a value suitable for
    minimization using an optimization algorithm.

    Parameters
    ----------
    xpars : np.ndarray
        Array of parameters to be fitted, typically including location, scale, shape,
        and possibly a second shape parameter and correction terms for covariates.
    data : np.ndarray
        Observed data points (e.g., annual maximum precipitation values).
    fit_kappa : bool
        If True, fit a Kappa-4 distribution (which includes a second shape parameter).
        If False, fit a GEV distribution.
    n_model : int
        Model number indicating how to derive parameters (location, scale) from
        given variables.
    ok_use_cov2 : bool
        If True, use a second covariate (cov2) in parameter modeling.
    n_cov2_term_option : int
        Option controlling how covariate 2 is incorporated into the model.
    mam : np.ndarray
        Mean Annual Maximum values for each data point.
    map : np.ndarray
        Mean Annual Precipitation values for each data point.
    cov1 : np.ndarray
        Covariate 1 values.
    cov2 : np.ndarray
        Covariate 2 values.
    w : np.ndarray
        Array of spatial weights for each data point. Weights should be consistent
        per station.
    shp : float
        Fixed shape parameter, if `ok_fix_shp` is True.
    shp2 : float
        Fixed second shape parameter, if `fit_kappa` and `ok_fix_shp` are True.
    ok_fix_shp : bool
        If True, use `shp` and `shp2` as fixed shape parameters rather than deriving
        them from `xpars`.

    Returns
    -------
    float
        The negative of the weighted log-likelihood. Minimizing this value will maximize
        the likelihood.

    Notes
    -----
    - The function internally calls `get_4pars_from_xpars` to derive location, scale,
      and shape parameters.
    - The shape parameters can be overridden if `ok_fix_shp` is True.
    - Uses `kappa4.logpdf` if `fit_kappa` is True, otherwise `genextreme.logpdf`.
    - Weights `w` are applied multiplicatively to the log-likelihood contributions of
      each data point.
    """
    # Compute distribution parameters
    loc, scl, shape, shape2 = get_4pars_from_xpars(
        xpars, fit_kappa, n_model, ok_use_cov2, n_cov2_term_option, mam, map, cov1, cov2
    )

    # Override shape parameters if they are fixed
    if ok_fix_shp:
        shape = shp
        shape2 = shp2 if fit_kappa else 0.0

    # Compute negative weighted log-likelihood
    if fit_kappa:
        # Kappa-4 distribution
        m_mle = -np.sum(kappa4.logpdf(data, shape2, shape, loc=loc, scale=scl) * w)
    else:
        # GEV distribution
        m_mle = -np.sum(genextreme.logpdf(data, shape, loc=loc, scale=scl) * w)

    return m_mle


def get_4pars_from_xpars(
    xpars: np.ndarray,
    fit_kappa: bool,
    n_model: int,
    ok_use_cov2: bool,
    n_cov2_term_option: int,
    mam: np.ndarray,
    map: np.ndarray,
    cov1: np.ndarray,
    cov2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute location, scale, and shape parameters from fitted parameters (xpars).

    This function derives the location (`loc`), scale (`scl`), and shape parameters
    (`shape` and optionally `shape2` if `fit_kappa` is True) for a given model based
    on the initial parameter vector `xpars` and the configuration of the model and
    covariates. The model determines how `loc` and `scl` are computed from `MAM`, `MAP`,
    and covariates. If covariates are used (`ok_use_cov2`), additional correction terms
    are applied.

    Parameters
    ----------
    xpars : np.ndarray
        Array of fitted parameters. The structure of xpars depends on:
        - If `fit_kappa` is True, then `xpars[3]` is shape2; else shape2 is assumed 0.
        - Depending on the model and covariates, additional correction terms may follow.
    fit_kappa : bool
        If True, indicates that the Kappa-4 distribution is fitted and `shape2` is
        included.
    n_model : int
        Model number indicating how location and scale parameters are derived:
        - 0: loc = xpars[0], scl = xpars[1]
        - 1: loc = xpars[0]*MAM, scl = xpars[1]*loc
        - 2: loc and scl proportional to MAM
        - 3: loc and scl proportional to MAP
        - 21: Starts from model 2 but corrects for one covariate (multiplicative)
        - 22: Starts from model 2 but corrects for one covariate (additive)
        - 4: loc and scl corrected by one covariate (additive)
    ok_use_cov2 : bool
        If True, a second covariate (cov2) is used to further adjust `loc` and `scl`.
    n_cov2_term_option : int
        Option for how covariate 2 modifies `loc` and `scl`:
        - 0: Additive correction
        - 1: Multiplicative correction
    mam : np.ndarray
        Mean Annual Maximum values.
    map : np.ndarray
        Mean Annual Precipitation values.
    cov1 : np.ndarray
        Covariate 1 values.
    cov2 : np.ndarray
        Covariate 2 values.

    Returns
    -------
    loc : np.ndarray
        Computed location parameter values.
    scl : np.ndarray
        Computed scale parameter values.
    shape : float
        Shape parameter.
    shape2 : float
        Second shape parameter (for Kappa-4 distribution). If `fit_kappa` is False,
        returns 0.0.

    Notes
    -----
    - Additional correction terms may follow the basic parameters
      (e.g., for n_model=21, 22, or 4) and for covariate 2 if `ok_use_cov2` is True.
    - Parameters are protected from unrealistic values by enforcing a minimum threshold
      (0.01) on `loc` and `scl`.
    """

    # Determine the starting index for correction terms
    # If Kappa-4 is fitted, shape2 is at xpars[3], so correction terms start at 4
    # Otherwise, shape2 is not fitted and correction terms start at 3
    if fit_kappa:
        kc = 4
    else:
        kc = 3

    # Compute loc and scl based on the selected model
    if n_model == 1:
        # loc proportional to MAM and scl proportional to loc
        loc = xpars[0] * mam
        scl = xpars[1] * loc
    elif n_model == 2:
        # loc and scl proportional to MAM
        loc = xpars[0] * mam
        scl = xpars[1] * mam
    elif n_model == 21:
        # Model 2 with multiplicative correction by cov1
        loc = xpars[0] * mam * (1.0 + xpars[kc] * cov1)
        scl = xpars[1] * mam * (1.0 + xpars[kc + 1] * cov1)
        kc += 2
    elif n_model == 22:
        # Model 2 with additive correction by cov1
        loc = xpars[0] * mam + xpars[kc] * cov1
        scl = xpars[1] * mam + xpars[kc + 1] * cov1
        kc += 2
    elif n_model == 3:
        # loc and scl proportional to MAP
        loc = xpars[0] * map
        scl = xpars[1] * map
    elif n_model == 4:
        # loc and scl corrected additively by cov1
        loc = xpars[0] + xpars[kc] * cov1
        scl = xpars[1] + xpars[kc + 1] * cov1
        kc += 2
    else:
        # n_model == 0 or fallback: default uniform parameters
        loc = xpars[0]
        scl = xpars[1]

    # Extract shape parameters
    shape = xpars[2]
    shape2 = xpars[3] if fit_kappa else 0.0

    # Apply covariate 2 corrections if used
    if ok_use_cov2:
        if n_cov2_term_option == 1:
            # Multiplicative correction with cov2
            loc = loc * (1.0 + xpars[kc] * cov2)
            scl = scl * (1.0 + xpars[kc + 1] * cov2)
        else:
            # Additive correction with cov2 (default)
            loc = loc + xpars[kc] * cov2
            scl = scl + xpars[kc + 1] * cov2

    # Protect for unreasonable loc and scl values
    tol = 0.01
    loc = np.where(loc < tol, tol, loc)
    scl = np.where(scl < tol, tol, scl)

    return loc, scl, shape, shape2


def get_distribution_parameters(
    xpars: np.ndarray,
    fit_kappa: bool,
    n_model: int,
    ok_use_cov2: bool,
    n_cov2_term_option: int,
    mam: np.ndarray,
    map_values: np.ndarray,
    cov1: np.ndarray,
    cov2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute location, scale, and shape parameters from fitted parameters.

    Parameters
    ----------
    xpars : np.ndarray
        Array of fitted parameters.
    fit_kappa : bool
        Indicates if the Kappa-4 distribution is fitted
        (i.e., shape2 parameter is included).
    n_model : int
        Model number determining how location and scale are calculated.
    ok_use_cov2 : bool
        Flag indicating whether to use covariate 2.
    n_cov2_term_option : int
        Option for how covariate 2 is applied (0 for additive, 1 for multiplicative).
    mam : np.ndarray
        Mean Annual Maximum (MAM) values.
    map_values : np.ndarray
        Mean Annual Precipitation (MAP) values.
    cov1 : np.ndarray
        Covariate 1 values.
    cov2 : np.ndarray
        Covariate 2 values.

    Returns
    -------
    loc : np.ndarray
        Computed location parameter values.
    scl : np.ndarray
        Computed scale parameter values.
    shape : float
        Shape parameter.
    shape2 : float
        Second shape parameter (for Kappa-4 distribution).

    Notes
    -----
    - The function calculates the location and scale parameters based on the
      selected model.
    - Covariates are applied according to specified options.
    - The function ensures that location and scale parameters are not less than a minimum
      threshold.
    """
    # Determine the starting index for correction terms
    if fit_kappa:
        kc = 4  # Shape2 parameter is included
    else:
        kc = 3  # Shape2 parameter is not included

    # Calculate location (loc) and scale (scl) based on the selected model
    if n_model == 1:
        # loc proportional to MAM and scl proportional to loc
        loc = xpars[0] * mam
        scl = xpars[1] * loc
    elif n_model == 2:
        # loc and scl proportional to MAM
        loc = xpars[0] * mam
        scl = xpars[1] * mam
    elif n_model == 21:
        # Model 2 corrected with one covariate (multiplicative correction)
        loc = xpars[0] * mam * (1.0 + xpars[kc] * cov1)
        scl = xpars[1] * mam * (1.0 + xpars[kc + 1] * cov1)
        kc += 2  # Update index for next term
    elif n_model == 22:
        # Model 2 corrected with one covariate (additive correction)
        loc = xpars[0] * mam + xpars[kc] * cov1
        scl = xpars[1] * mam + xpars[kc + 1] * cov1
        kc += 2
    elif n_model == 3:
        # loc and scl proportional to MAP
        loc = xpars[0] * map_values
        scl = xpars[1] * map_values
    elif n_model == 4:
        # loc and scl are constants corrected with covariate 1
        loc = xpars[0] + xpars[kc] * cov1
        scl = xpars[1] + xpars[kc + 1] * cov1
        kc += 2
    else:
        # Default model (n_model == 0): uniform parameters
        loc = xpars[0]
        scl = xpars[1]

    # Extract shape parameters
    shape = xpars[2]
    if fit_kappa:
        shape2 = xpars[3]
    else:
        shape2 = 0.0

    # Apply covariate 2 if used
    if ok_use_cov2:
        if n_cov2_term_option == 1:
            # Multiplicative term with covariate 2
            loc = loc * (1.0 + xpars[kc] * cov2)
            scl = scl * (1.0 + xpars[kc + 1] * cov2)
        else:
            # Additive term with covariate 2
            loc = loc + xpars[kc] * cov2
            scl = scl + xpars[kc + 1] * cov2

    # Ensure loc and scl are not less than the minimum allowed value
    tol = 0.01  # Minimum allowed value
    loc = np.where(loc < tol, tol, loc)
    scl = np.where(scl < tol, tol, scl)

    return loc, scl, shape, shape2


# get standardized variable from precip
def get_x_from_precip(precip, loc, scl):
    return (precip - loc) / scl


# get return period T from distribution for specified precip
def get_return_period_from_precip(precip, loc, scl, shape, shape2, fit_kappa):
    if fit_kappa:
        sf = kappa4.sf(
            precip, shape2, shape, loc=loc, scale=scl
        )  # sf: Survival function (1-cdf)
    else:
        sf = genextreme.sf(
            precip, shape, loc=loc, scale=scl
        )  # sf: Survival function (1-cdf)
    sf[sf < 1e-30] = 1e-30  # to avoid division by zero
    return 1.0 / sf  # 1/sf: return period from distribution


# expected return period or recurrence interval for sorted AMS values
def get_expected_return_period_from_nams(nams):
    return (nams + 1.0) / (
        nams + 1.0 - (np.arange(nams) + 1.0)
    )  # return period from observed values


# get precip from distribution for specified return period T
def get_precip_from_return_period(return_period, loc, scl, shape, shape2, fit_kappa):
    cdf = (
        1.0 - 1.0 / return_period
    )  # converting return period to cumulative distribution function value
    if fit_kappa:
        ppf = kappa4.ppf(
            cdf, shape2, shape, loc=loc, scale=scl
        )  # ppf: Percent point function (inverse of cdf — percentiles).
    else:
        ppf = genextreme.ppf(
            cdf, shape, loc=loc, scale=scl
        )  # ppf: Percent point function (inverse of cdf — percentiles).
    return ppf  # precipitation. When loc=0 and scl=1, return standardized value


# get precip from standardized variable
def get_precip_from_x(x, loc, scl):
    return x * scl + loc


# Estimation of Akaike’s Information Criterion (AIC) and
# Bayesian information criterion (bic).
# Corrected AIC (cor_aic) can be used also for small sample sizes.
# Better model produces smaller AIC and bic.
# from Multimodel Inference Understanding AIC and bic in Model Selection
# by KENNETH P. BURNHAM, DAVID R. ANDERSON at Colorado Cooperative Fish and
# Wildlife Research Unit (USGS-BRD)
# m_mle: negative MLE resulting from function nopenlik. Notice that this is
# before dividing by sum(w)
# nparams: number of fitted parameters; ndata: number of AMS values, but in this case,
# sum(w) can be used as ndata.
# To compare among different models, compute exp(cor_aic0-cor_aic), where 1 is from the
# best model with cor_aic0
def get_cor_aic_bic(m_mle, nparams, ndata):
    cor_aic = 2.0 * (
        m_mle + nparams * (1.0 + (nparams + 1.0) / (ndata - nparams - 1.0))
    )  # Akaike's information criterion - corrected for small samples
    bic = 2.0 * m_mle + nparams * np.log(ndata)  # Bayesian information criterion
    return cor_aic, bic


def create_error_plot(
    x: np.ndarray,
    xt: np.ndarray,
    expected_return_period: np.ndarray,
    errors: Tuple[float, float, float],
    shape_param: float,
    title: str,
    output_path: Path,
) -> None:
    """Create and save plot comparing fitted vs observed distribution values.

    Args:
        x: Observed standardized values
        xt: Fitted standardized values
        expected_return_period: Return periods
        errors: Tuple of (mean_error, mean_abs_error, rmse)
        shape_param: Shape parameter of the distribution
        title: Plot title
        output_path: Directory path to save the plot

    Raises:
        Exception: If there's an error in plot creation or saving
    """
    try:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        df = pd.DataFrame({"x": x, "xt": xt, "Te": expected_return_period})
        df.sort_values(by=["Te"], inplace=True)

        ax.scatter(df["Te"], df["x"], marker=".", label="Observed")
        ax.plot(df["Te"], df["xt"], label="Fitted")

        plt.xlabel("Return Period (Year)")
        plt.ylabel("Annual Maximum Precip - standardized variable")

        me, mae, rmse = errors
        plt.title(
            f"{title}, shape={shape_param:.3f}\n"
            f"nAMS={len(x)}, ME={me:.3f}, "
            f"MAE={mae:.3f}, RMSE={rmse:.3f}"
        )

        plt.legend()
        plt.xscale("log")
        ax.grid(which="major", linewidth=0.5, linestyle="dashed", color="gray")
        ax.grid(which="minor", linewidth=0.1, linestyle="dashed", color="gray")

        output_path.mkdir(exist_ok=True)
        fig.savefig(output_path / f"{title}.png", bbox_inches="tight", dpi=300)
        plt.close(fig)

    except Exception as e:
        logger.error(f"Error creating plot: {str(e)}")
        raise


def get_fit_distribution_errors(
    x: np.ndarray,
    xt: np.ndarray,
    expected_return_period: np.ndarray,
    w: np.ndarray,
    scl0: float,
    shp0: float,
    ok_errors_in_inches: bool,
    ok_save_plot: bool,
    title_plot: str,
    output_path: Path,
) -> Tuple[float, float, float]:
    """Calculate fitting errors between observed and fitted distribution values.

    Args:
        x: Observed standardized values
        xt: Fitted standardized values
        expected_return_period: Return periods
        w: Weights for each observation
        scl0: Scale parameter for converting to inches
        shp0: Shape parameter of the distribution
        ok_errors_in_inches: If True, convert errors to inches
        ok_save_plot: If True, save comparison plot
        title_plot: Title for the plot
        output_path: Directory path to save the plot

    Returns:
        Tuple containing:
            - me: Weighted mean error
            - mae: Weighted mean absolute error
            - rmse: Weighted root mean squared error

    Raises:
        Exception: If there's an error in calculations or plotting
    """
    try:
        dx = x - xt
        sum_w = np.sum(w)

        me = np.sum(w * dx) / sum_w
        mae = np.sum(w * np.abs(dx)) / sum_w
        rmse = np.sqrt(np.sum(w * dx * dx) / sum_w)

        if ok_save_plot:
            create_error_plot(
                x,
                xt,
                expected_return_period,
                (me, mae, rmse),
                shp0,
                title_plot,
                output_path,
            )

        if ok_errors_in_inches:
            me *= scl0
            mae *= scl0
            rmse *= scl0

        return me, mae, rmse

    except Exception as e:
        logger.error(f"Error calculating distribution errors: {str(e)}")
        raise


def create_fitting_plot(
    sub_ams: pd.DataFrame,
    id_numbers: List[int],
    shape_param: float,
    errors: Tuple[float, float, float],
    plot_in_inches: bool,
    title: str,
    output_path: Path,
) -> None:
    """Create and save plot comparing regional fitted vs observed data.

    Args:
        sub_ams: DataFrame containing AMS data with columns
        ['Te', 'precip', 'pt', 'x', 'xt', 'id_num']
        id_numbers: List of station IDs to plot
        shape_param: Shape parameter value for the distribution
        errors: Tuple of (mean_error, mean_abs_error, rmse)
        plot_in_inches: If True, plot in inches; if False, plot standardized values
        title: Title for the plot
        output_path: Directory path to save the plot

    Raises:
        Exception: If there's an error in plot creation or saving
    """
    try:
        fig, ax = plt.subplots(nrows=1, ncols=1)

        if plot_in_inches:
            ax.scatter(
                sub_ams["Te"], sub_ams["precip"], marker=".", label="Regional Observed"
            )
            ax.plot(sub_ams["Te"], sub_ams["pt"], label="Regional Fitted")
        else:
            ax.scatter(sub_ams["Te"], sub_ams["x"], marker=".", label="Regional Observed")
            ax.plot(sub_ams["Te"], sub_ams["xt"], label="Regional Fitted")

        for station_id in id_numbers:
            station_data = sub_ams[sub_ams["id_num"] == station_id]
            if len(station_data) > 0:
                expected_return_period = get_expected_return_period_from_nams(
                    len(station_data)
                )
                if plot_in_inches:
                    ax.scatter(
                        expected_return_period,
                        station_data.precip.values,
                        marker=".",
                        label=f"Sta ID={station_id}",
                    )
                else:
                    ax.scatter(
                        expected_return_period,
                        station_data.x.values,
                        marker=".",
                        label=f"Sta ID={station_id}",
                    )

        plt.xlabel("Return Period (Year)")
        plt.ylabel(
            "Annual Maximum Precip (inches)"
            if plot_in_inches
            else "Annual Maximum Precip - standardized variable"
        )

        me, mae, rmse = errors
        plt.title(
            f"{title}, shape={shape_param:.3f}\n"
            f"nAMS={len(sub_ams)}, ME={me:.3f}, "
            f"MAE={mae:.3f}, RMSE={rmse:.3f}"
        )

        plt.legend()
        plt.xscale("log")
        ax.grid(which="major", linewidth=0.5, linestyle="dashed", color="gray")
        ax.grid(which="minor", linewidth=0.1, linestyle="dashed", color="gray")

        output_path.mkdir(exist_ok=True)
        fig.savefig(output_path / f"{title}.png", bbox_inches="tight", dpi=300)
        plt.close(fig)

    except Exception as e:
        logger.error(f"Error creating plot: {str(e)}")
        raise


def get_fit_distribution_errors_station(
    sub_ams_in: pd.DataFrame,
    id_num: List[int],
    fit_kappa: bool,
    scl0: float,
    shp0: float,
    ok_errors_in_inches: bool,
    ok_save_plot: bool,
    title_plot: str,
    ok_plot_in_inch: bool,
    output_path: Path,
) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate fitting errors of standardized variables by station.

    Args:
        sub_ams_in: DataFrame containing AMS data with required columns
            ['id_num', 'x', 'w', 'shape', 'shape2', 'scl']
        id_num: List of station IDs to analyze
        fit_kappa: Boolean flag for k4 fitting method
        scl0: Scale parameter for the region center
        shp0: Shape parameter for the region center
        ok_errors_in_inches: If True, convert errors to inches
        ok_save_plot: If True, save comparison plot
        title_plot: Title for the plot
        ok_plot_in_inch: If True, plot in inches
        output_path: Directory path to save the plot

    Returns:
        Tuple containing:
            - mean_error: Regional weighted mean error
            - mean_abs_error: Regional weighted mean absolute error
            - rmse: Regional weighted root mean squared error
            - me_array: Array of mean errors for each station
            - mae_array: Array of mean absolute errors for each station
            - rmse_array: Array of root mean squared errors for each station

    Raises:
        ValueError: If sum of weights is zero
        Exception: For other calculation or plotting errors
    """
    try:
        ns = len(id_num)
        sum_w = 0.0
        me = 0.0
        mae = 0.0
        rmse = 0.0
        me_array = np.zeros(ns)
        mae_array = np.zeros(ns)
        rmse_array = np.zeros(ns)

        for idx, station_id in enumerate(id_num):
            station_data = sub_ams_in[sub_ams_in["id_num"] == station_id]

            if len(station_data) > 0 and station_data.w.values.sum() > 0:
                expected_return_period = get_expected_return_period_from_nams(
                    len(station_data)
                )
                xt1 = get_precip_from_return_period(
                    expected_return_period,
                    0.0,
                    1.0,
                    station_data["shape"].values,
                    station_data.shape2.values,
                    fit_kappa,
                )

                me_array[idx], mae_array[idx], rmse_array[idx] = (
                    get_fit_distribution_errors(
                        station_data.x.values,
                        xt1,
                        expected_return_period,
                        station_data.w.values,
                        1.0,
                        0.0,
                        False,
                        False,
                        "",
                        "",
                    )
                )

                sum_wk = station_data.w.sum()
                sum_w += sum_wk
                me += sum_wk * me_array[idx]
                mae += sum_wk * mae_array[idx]
                rmse += sum_wk * rmse_array[idx] ** 2

                if ok_errors_in_inches:
                    me_array[idx] *= station_data.scl.values[0]
                    mae_array[idx] *= station_data.scl.values[0]
                    rmse_array[idx] *= station_data.scl.values[0]

        if sum_w == 0:
            raise ValueError("Sum of weights is zero")

        me /= sum_w
        mae /= sum_w
        rmse = np.sqrt(rmse / sum_w)

        if ok_save_plot:
            create_fitting_plot(
                sub_ams_in,
                id_num,
                shp0,
                (me, mae, rmse),
                ok_plot_in_inch,
                title_plot,
                output_path,
            )

        if ok_errors_in_inches:
            me *= scl0
            mae *= scl0
            rmse *= scl0

        return me, mae, rmse, me_array, mae_array, rmse_array

    except Exception as e:
        logger.error(f"Error in fitting distribution: {str(e)}")
        raise
