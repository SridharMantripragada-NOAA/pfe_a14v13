# statistical_functions.py

import logging
import math
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import (
    chisquare,
    genextreme,
    kappa4,
    ks_1samp,
    ks_2samp,
)

logger = logging.getLogger(__name__)


def get_kstest_1samp(
    x: np.ndarray,
    shape: float,
    shape2: float,
    ok_fit_k4: bool,
    pvalue_min: float,
) -> Tuple[bool, float]:
    """
    Perform a one-sample Kolmogorov-Smirnov test for goodness of fit.

    The null hypothesis is that the sample comes from the given distribution.

    Parameters
    ----------
    x : np.ndarray
        Standardized sample data.
    shape : float
        Shape parameter of the distribution.
    shape2 : float
        Second shape parameter (used for kappa4 distribution).
    ok_fit_k4 : bool
        If True, use kappa4 distribution; otherwise, use GEV (genextreme).
    pvalue_min : float
        Significance level (e.g., 0.05).

    Returns
    -------
    ok_ks_rejected : bool
        True if the null hypothesis is rejected.
    pvalue : float
        The p-value of the test.

    Notes
    -----
    The test is done on standardized values and not on the original data.
    This test ignores the spatial weighting used to fit the distribution parameters.
    """
    if not isinstance(x, np.ndarray) or x.size == 0:
        raise ValueError("Input array x must be a non-empty numpy array.")

    if not isinstance(shape, (float, int)):
        raise ValueError("Shape parameter 'shape' must be a float.")

    if ok_fit_k4 and not isinstance(shape2, (float, int)):
        raise ValueError(
            "Shape parameter 'shape2' must be a float when 'ok_fit_k4' is True."
        )

    try:
        if ok_fit_k4:
            res = ks_1samp(x, kappa4.cdf, args=(shape2, shape))
            # No need for bias correction as found from Monte Carlo simulations.
        else:
            res = ks_1samp(x, genextreme.cdf, args=(shape,))
        ok_ks_rejected = res.pvalue < pvalue_min  # True when null hypothesis is rejected.
        return ok_ks_rejected, res.pvalue
    except Exception as e:
        logger.exception("Error in get_kstest_1samp: %s", e)
        raise


def get_kstest_2samp(
    x1: np.ndarray,
    x2: np.ndarray,
    pvalue_min: float = 0.05,
) -> Tuple[bool, float]:
    """
    Perform a two-sample Kolmogorov-Smirnov test.

    The null hypothesis is that the two samples come from the same distribution.

    Parameters
    ----------
    x1 : np.ndarray
        First sample data.
    x2 : np.ndarray
        Second sample data.
    pvalue_min : float, optional
        Significance level (default is 0.05).

    Returns
    -------
    ok_ks_rejected : bool
        True if the null hypothesis is rejected.
    pvalue : float
        The adjusted p-value of the test.

    Notes
    -----
    A bias correction is applied to the p-value as found from Monte Carlo simulations.
    The correction is valid in the p-value range [0.01, 0.25].
    """
    if not isinstance(x1, np.ndarray) or x1.size == 0:
        raise ValueError("Input array x1 must be a non-empty numpy array.")
    if not isinstance(x2, np.ndarray) or x2.size == 0:
        raise ValueError("Input array x2 must be a non-empty numpy array.")

    try:
        res = ks_2samp(x1, x2)
        # Bias correction as found from Monte Carlo simulations.
        pvalue = (
            res.pvalue * 1.0227
        ) ** 1.1124  # Correction valid in p-value range [0.01, 0.25]
        ok_ks_rejected = pvalue < pvalue_min  # True when null hypothesis is rejected.
        return ok_ks_rejected, pvalue
    except Exception as e:
        logger.exception("Error in get_kstest_2samp: %s", e)
        raise


def get_chi2test_2samp(
    x1: np.ndarray,
    x2: np.ndarray,
    pvalue_min: float,
    ok_debug: bool = False,
) -> Tuple[bool, float, int]:
    """
    Perform a Chi-squared test comparing two samples.

    The null hypothesis is that the two sets of values come from the same distribution.

    Parameters
    ----------
    x1 : np.ndarray
        First sample data.
    x2 : np.ndarray
        Second sample data.
    pvalue_min : float
        Significance level (e.g., 0.05).
    ok_debug : bool, optional
        If True, debug information is logged (default is False).

    Returns
    -------
    ok_chi2_rejected : bool
        True if the null hypothesis is rejected.
    pvalue : float
        The p-value of the test.
    num_groups : int
        Number of groups formed in the test.

    Notes
    -----
    This implementation groups the data into intervals to have at least `nmin=5`
    expected elements.
    The grouping procedure starts at min(x) and tends to produce a larger group for the
    highest x interval.
    Therefore, the test could be repeated by entering `-x1` and `-x2` to have different
    groups.
    """
    nmin = 5  # Minimum number of expected elements in each class for the test to be valid

    if not isinstance(x1, np.ndarray) or x1.size == 0:
        raise ValueError("Input array x1 must be a non-empty numpy array.")
    if not isinstance(x2, np.ndarray) or x2.size == 0:
        raise ValueError("Input array x2 must be a non-empty numpy array.")

    try:
        x1s = np.sort(x1)  # Sorted dataset 1
        n1 = len(x1s)
        x2s = np.sort(x2)  # Sorted dataset 2
        n2 = len(x2s)

        if n1 < n2:
            ngmin = math.ceil(nmin * (n1 + n2) / n1)
        else:
            ngmin = math.ceil(nmin * (n1 + n2) / n2)

        xlim = []  # Interval limits
        ng1 = []  # Number of elements in interval for x1
        ng2 = []  # Number of elements in interval for x2
        k1 = -1
        k2 = -1

        while (k1 < n1 - 1) and (k2 < n2 - 1) and (n1 - k1 + n2 - k2 > 2):
            k10 = k1
            k20 = k2
            ngk = (k1 - k10) + (k2 - k20)

            while (ngk < ngmin) and (k1 + 1 < n1) and (k2 + 1 < n2):
                if x1s[k1 + 1] < x2s[k2 + 1]:
                    k1 += 1
                    xmax = x1s[k1]
                else:
                    k2 += 1
                    xmax = x2s[k2]
                # Update index 1
                while (k1 + 1 < n1) and (x1s[k1 + 1] <= xmax):
                    k1 += 1
                if k1 + 1 == n1:
                    k2 = n2 - 1  # Take remaining elements of sample 2
                    xmax = max(xmax, x2s[k2])
                # Update index 2
                while (k2 + 1 < n2) and (x2s[k2 + 1] <= xmax):
                    k2 += 1
                if k2 + 1 == n2:
                    k1 = n1 - 1  # Take remaining elements of sample 1
                    xmax = max(xmax, x1s[k1])
                ngk = (k1 - k10) + (k2 - k20)

            if ngk > ngmin * 0.95:
                # Add new group
                xlim.append(xmax)
                ng1.append(k1 - k10)
                ng2.append(k2 - k20)
            else:
                # Take remaining elements
                k1 = n1 - 1
                k2 = n2 - 1
                xmax = max(x1s[k1], x2s[k2])
                if not xlim:
                    # Not a group yet, then create a new group
                    xlim.append(xmax)
                    ng1.append(k1 - k10)
                    ng2.append(k2 - k20)
                else:
                    # Add elements to last group
                    xlim[-1] = xmax
                    ng1[-1] += k1 - k10
                    ng2[-1] += k2 - k20

        if len(xlim) < 2:
            # Could not make at least two groups to perform the test. Test aborted
            return False, float("nan"), 0
        # Create dataframe
        df = pd.DataFrame(
            {
                "xlim": xlim,
                "ng1": ng1,
                "ng2": ng2,
            }
        )
        df["ng"] = df["ng1"] + df["ng2"]
        # Compute expected values
        fng = df["ng"] / df["ng"].sum()
        df["f_exp1"] = fng * df["ng1"].sum()
        df["f_exp2"] = fng * df["ng2"].sum()

        if ok_debug or (df["f_exp1"].min() < nmin - 1) or (df["f_exp2"].min() < nmin - 1):
            logger.debug("In get_chi2test_2samp:\n%s", df)

        # Reduction in degrees of freedom
        ddof = len(df)
        # Chi-squared test for all cells
        res = chisquare(
            np.append(df["ng1"].values, df["ng2"].values),
            f_exp=np.append(df["f_exp1"].values, df["f_exp2"].values),
            ddof=ddof,
        )
        ok_chi2_rejected = (
            res.pvalue < pvalue_min
        )  # True when null hypothesis is rejected.

        if ok_debug:
            logger.debug(
                "pvalue=%s, rejected?=%s", round(res.pvalue, 3), ok_chi2_rejected
            )

        return ok_chi2_rejected, res.pvalue, len(df)
    except Exception as e:
        logger.exception("Error in get_chi2test_2samp: %s", e)
        raise


def get_chi2test_2samp_ni(
    x1: np.ndarray,
    x2: np.ndarray,
    ni: int,
    pvalue_min: float,
    ok_debug: bool = False,
) -> Tuple[bool, float, int]:
    """
    Perform a Chi-squared test comparing two samples, grouping data evenly into
    intervals.

    The null hypothesis is that the two sets of values come from the same distribution.

    Parameters
    ----------
    x1 : np.ndarray
        First sample data.
    x2 : np.ndarray
        Second sample data.
    ni : int
        Number of intervals to group data.
    pvalue_min : float
        Significance level (e.g., 0.05).
    ok_debug : bool, optional
        If True, debug information is logged (default is False).

    Returns
    -------
    ok_chi2_rejected : bool
        True if the null hypothesis is rejected.
    pvalue : float
        The p-value of the test.
    num_groups : int
        Number of groups formed in the test.

    Notes
    -----
    This implementation groups the data evenly into `ni` intervals.
    Ensures that intervals have at least `nmin=5` expected elements on average.
    """
    nmin = 5  # Minimum number of expected elements in each class for the test to be valid

    if not isinstance(x1, np.ndarray) or x1.size == 0:
        raise ValueError("Input array x1 must be a non-empty numpy array.")
    if not isinstance(x2, np.ndarray) or x2.size == 0:
        raise ValueError("Input array x2 must be a non-empty numpy array.")

    try:
        if (len(x1) < 2 * nmin) or (len(x2) < 2 * nmin):
            # Could not make at least two groups with at least nmin=5 expected elements
            return False, float("nan"), 0

        if len(x1) / ni < nmin:
            ni = len(x1) // nmin
        if len(x2) / ni < nmin:
            ni = len(x2) // nmin
        if ni < 2:
            ni = 2  # Minimum number of intervals

        x12 = np.sort(np.concatenate((x1, x2)))  # Merged and sorted array
        n12 = len(x12)

        # Find interval limits and count elements in groups
        xlim = []  # Interval limits
        ng1 = []  # Number of elements in interval for x1
        ng2 = []  # Number of elements in interval for x2
        na10 = 0  # Number of elements lower than given xlim in x1
        na20 = 0  # Number of elements lower than given xlim in x2

        for ki in range(ni):
            idx = ((n12 - 1) * (ki + 1)) // ni
            xlim.append(x12[idx])  # Interval limit
            na1 = np.sum(x1 <= xlim[ki])  # Elements in x1 lower than xlim
            na2 = np.sum(x2 <= xlim[ki])  # Elements in x2 lower than xlim
            ng1.append(na1 - na10)  # Elements in current interval for x1
            ng2.append(na2 - na20)  # Elements in current interval for x2
            na10 = na1
            na20 = na2

        # Create dataframe
        df = pd.DataFrame(
            {
                "xlim": xlim,
                "ng1": ng1,
                "ng2": ng2,
            }
        )
        df["ng"] = df["ng1"] + df["ng2"]
        # Compute expected values
        fng = df["ng"] / df["ng"].sum()
        df["f_exp1"] = fng * df["ng1"].sum()
        df["f_exp2"] = fng * df["ng2"].sum()

        if ok_debug or (df["f_exp1"].min() < nmin - 1) or (df["f_exp2"].min() < nmin - 1):
            logger.debug("In get_chi2test_2samp_ni:\n%s", df)

        # Reduction in degrees of freedom
        ddof = len(df)
        # Chi-squared test
        res = chisquare(
            np.concatenate((df["ng1"].values, df["ng2"].values)),
            f_exp=np.concatenate((df["f_exp1"].values, df["f_exp2"].values)),
            ddof=ddof,
        )
        ok_chi2_rejected = (
            res.pvalue < pvalue_min
        )  # True when null hypothesis is rejected.

        if ok_debug:
            logger.debug(
                "pvalue=%s, rejected?=%s", round(res.pvalue, 3), ok_chi2_rejected
            )

        return ok_chi2_rejected, res.pvalue, len(df)
    except Exception as e:
        logger.exception("Error in get_chi2test_2samp_ni: %s", e)
        raise


def get_chi2test_1samp_ni(
    x: np.ndarray,
    shape: float,
    shape2: float,
    ok_fit_k4: bool,
    ni: int,
    pvalue_min: float,
    ok_debug: bool = False,
) -> Tuple[bool, float, int]:
    """
    Perform a Chi-squared test comparing a sample to a distribution, grouping data into
    intervals.

    The null hypothesis is that the sample comes from the given distribution.

    Parameters
    ----------
    x : np.ndarray
        Sample data.
    shape : float
        Shape parameter of the distribution.
    shape2 : float
        Second shape parameter (used for kappa4 distribution).
    ok_fit_k4 : bool
        If True, use kappa4 distribution; otherwise, use GEV (genextreme).
    ni : int
        Number of intervals to group data.
    pvalue_min : float
        Significance level (e.g., 0.05).
    ok_debug : bool, optional
        If True, debug information is logged (default is False).

    Returns
    -------
    ok_chi2_rejected : bool
        True if the null hypothesis is rejected.
    pvalue : float
        The p-value of the test.
    num_groups : int
        Number of groups formed in the test.

    Notes
    -----
    Ensures that intervals have at least `nmin=5` expected elements on average.
    """
    nmin = 5  # Minimum number of expected elements in each class for the test to be valid

    if not isinstance(x, np.ndarray) or x.size == 0:
        raise ValueError("Input array x must be a non-empty numpy array.")

    if not isinstance(shape, (float, int)):
        raise ValueError("Shape parameter 'shape' must be a float.")

    if ok_fit_k4 and not isinstance(shape2, (float, int)):
        raise ValueError(
            "Shape parameter 'shape2' must be a float when 'ok_fit_k4' is True."
        )

    try:
        nx = len(x)
        if nx < 2 * nmin:
            # Could not make at least two groups with at least nmin=5 elements
            return False, float("nan"), 0

        if nx / ni < nmin:
            ni = nx // nmin
        if ni < 2:
            ni = 2  # Minimum number of intervals

        # Expected cdf values
        step = 1.0 / ni
        cdf = np.arange(start=step, stop=0.999, step=step)

        # Find xlim values from distribution with location=0 and scale=1
        if ok_fit_k4:
            xlim = kappa4.ppf(cdf, shape2, shape)
        else:
            xlim = genextreme.ppf(cdf, shape)

        # Count elements in groups
        ng = []  # Number of elements in each interval
        na0 = 0  # Number of elements lower than given xlim

        for ki in range(len(xlim)):
            na = np.sum(x <= xlim[ki])
            ng.append(na - na0)
            na0 = na
        ng.append(nx - na0)  # Last interval

        # Reduction in degrees of freedom
        ddof = 0
        # Chi-squared test
        res = chisquare(ng, ddof=ddof)
        ok_chi2_rejected = (
            res.pvalue < pvalue_min
        )  # True when null hypothesis is rejected.

        if ok_debug:
            logger.debug("In get_chi2test_1samp_ni:")
            logger.debug("xlim=%s", xlim)
            logger.debug("ng=%s", ng)
            logger.debug(
                "pvalue=%s, rejected?=%s", round(res.pvalue, 3), ok_chi2_rejected
            )

        return ok_chi2_rejected, res.pvalue, len(ng)
    except Exception as e:
        logger.exception("Error in get_chi2test_1samp_ni: %s", e)
        raise


def differentiate_values(x: np.ndarray, epx: float) -> np.ndarray:
    """
    Differentiate equal values in an array by adding/subtracting negligible quantities.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    epx : float
        Small quantity to differentiate equal values.

    Returns
    -------
    y : np.ndarray
        Array with differentiated values.
    """
    if not isinstance(x, np.ndarray) or x.size == 0:
        raise ValueError("Input array x must be a non-empty numpy array.")

    y = np.sort(x)  # Sort array
    dy = np.zeros_like(y)
    nr = 0  # Number of repetitions
    ys = 1  # Sign for adding or subtracting

    for k in range(len(y) - 1):
        if y[k] == y[k + 1]:
            if (nr == 0) or (ys == 1):
                nr += 1  # Increment repetition count
            ys = -ys  # Change sign
            dy[k] = ys * nr * epx  # Small difference to add
        else:
            nr = 0
            ys = 1

    return y + dy


def find_chi2_ni(nx: int) -> int:
    """
    Find the number of intervals or groups for Chi-squared test.

    Parameters
    ----------
    nx : int
        Number of elements in the sample.

    Returns
    -------
    ni : int
        Number of intervals or groups.
    """
    nmin = 5  # Minimum number of expected elements in each class for the test to be valid
    nimin = 2  # Minimum number of groups to form
    nimax = 15  # Maximum number of groups to form

    ni = math.floor(math.sqrt(nx) / 1.1)
    if ni * nmin > nx:
        ni = nx // nmin
    if ni < nimin:
        ni = nimin
    elif ni > nimax:
        ni = nimax
    return ni


def get_chi2test_2samp_all(
    x1: np.ndarray,
    x2: np.ndarray,
    pvalue_min: float = 0.05,
    ok_debug: bool = False,
) -> Tuple[bool, float, float]:
    """
    Perform a Chi-squared test comparing two samples, trying different groupings.

    The null hypothesis is that the two sets of values come from the same distribution.

    Parameters
    ----------
    x1 : np.ndarray
        First sample data.
    x2 : np.ndarray
        Second sample data.
    pvalue_min : float, optional
        Significance level (default is 0.05).
    ok_debug : bool, optional
        If True, debug information is logged (default is False).

    Returns
    -------
    ok_chi2_rejected : bool
        True if the null hypothesis is rejected.
    pvalue : float
        The adjusted median p-value of the test.
    perc_failure : float
        Percentage of tests where the null hypothesis was rejected.

    Notes
    -----
    This implementation tries different ways to make the groups and reports the
    median p-value.
    Ensures that intervals have at least `nmin=5` expected elements on average.
    """
    nmin = 5  # Minimum number of expected elements in each class for the test to be valid

    if not isinstance(x1, np.ndarray) or x1.size == 0:
        raise ValueError("Input array x1 must be a non-empty numpy array.")
    if not isinstance(x2, np.ndarray) or x2.size == 0:
        raise ValueError("Input array x2 must be a non-empty numpy array.")

    try:
        if (len(x1) < 2 * nmin) or (len(x2) < 2 * nmin):
            # Could not make at least two groups with at least nmin=5 elements
            return False, float("nan"), 0.0

        y1 = differentiate_values(x1, 0.0001)
        y2 = differentiate_values(x2, 0.0001)

        # Collect results in a list
        results_list = []
        nx = min(len(x1), len(x2))
        nimax = find_chi2_ni(nx)

        for ni in range(nimax, 1, -1):
            # Original x values
            ok_chi2_rejected, pvalue, num_groups = get_chi2test_2samp_ni(
                y1, y2, ni, pvalue_min, ok_debug
            )
            if num_groups > 0:
                results_list.append(
                    {"xsign": 1, "ngroups": num_groups, "pvalue": round(pvalue, 5)}
                )

            # Negative x values
            ok_chi2_rejected, pvalue, num_groups = get_chi2test_2samp_ni(
                -y1, -y2, ni, pvalue_min, ok_debug
            )
            if num_groups > 0:
                results_list.append(
                    {"xsign": -1, "ngroups": num_groups, "pvalue": round(pvalue, 5)}
                )

        if results_list:
            df_results = pd.DataFrame(results_list)
            pvalue_median = df_results["pvalue"].median()
            # Bias correction
            pvalue_adjusted = (
                pvalue_median * 1.1193
            ) ** 1.2842  # Correction valid in p-value range [0.01, 0.25]
            perc_failure = (
                100.0 * (df_results["pvalue"] < pvalue_min).sum() / len(df_results)
            )
            ok_chi2_rejected = pvalue_adjusted < pvalue_min

            if ok_debug:
                logger.debug("In get_chi2test_2samp_all:\n%s", df_results)
                logger.debug(
                    "Median pvalue=%s, Adjusted pvalue=%s, Test failure=%.1f%%,"
                    "Rejected=%s",
                    pvalue_median,
                    pvalue_adjusted,
                    perc_failure,
                    ok_chi2_rejected,
                )

            return ok_chi2_rejected, pvalue_adjusted, perc_failure
        return False, float("nan"), 0.0
    except Exception as e:
        logger.exception("Error in get_chi2test_2samp_all: %s", e)
        raise
