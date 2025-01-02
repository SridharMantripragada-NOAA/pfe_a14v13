import logging
from pathlib import Path
from typing import Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_cov_from_map(map: np.ndarray, n_map_option: int) -> np.ndarray:
    """
    Compute the covariate from the Mean Annual Precipitation (MAP) using different
    transformation options.

    Parameters:
        MAP (np.ndarray): Array of Mean Annual Precipitation values.
        n_map_option (int): Option to transform MAP:
                          - 0: Normalized MAP (default)
                          - 1: Square root of normalized MAP
                          - 2: Cubic root of normalized MAP

    Returns:
        np.ndarray: Transformed covariate values, centered at zero.
    """
    # Default option: Normalize by using the overall median (16 inches for Volume 12)
    cov = map / 16.0

    if n_map_option == 1:
        cov = np.sqrt(cov)  # Square root transformation
    elif n_map_option == 2:
        cov = cov ** (1.0 / 3.0)  # Cubic root transformation

    return cov - 1.0  # Center covariate at zero


def get_cov_from_year(year: np.ndarray) -> np.ndarray:
    """
    Compute the covariate from the year, normalized and centered at zero.

    Parameters:
        year (np.ndarray): Array of year values.

    Returns:
        np.ndarray: Normalized and centered covariate values based on the year.
    """
    # Normalize the year using the range from pre-industrial (1960) to current (2024)
    cov = (year - 1960.0) / (2024.0 - 1960.0)

    return cov - 0.5  # Center covariate at zero


def get_cov_from_co2(co2: np.ndarray) -> np.ndarray:
    """
    Compute the covariate from CO2 concentrations using logarithmic scaling,
    normalized and centered at zero.

    Parameters:
        co2 (np.ndarray): Array of co2 concentration values (in ppm).

    Returns:
        np.ndarray: Normalized and centered covariate values based on co2 concentrations.
    """
    # Normalize using logarithmic scaling, with pre-industrial = 280 ppm and
    # current (2022) = 420 ppm
    cov = np.log(co2 / 280.0) / np.log(420.0 / 280.0)

    return cov - 0.5  # Center covariate at zero


def get_cov_from_mam(mam: np.ndarray) -> np.ndarray:
    """
    Compute the covariate from Mean Annual Maximum (MAM) precipitation,
    normalized and centered at zero.

    Parameters:
        mam (np.ndarray): Array of MAM values (in inches for a 1-day duration).

    Returns:
        np.ndarray: Normalized and centered covariate values based on MAM.
    """
    # Normalize by the overall median in Volume 12 (1.4 inches for 1-day MAM)
    cov = mam / 1.4
    return cov - 1.0  # Center covariate at zero


def get_cov_from_elev(elev: np.ndarray) -> np.ndarray:
    """
    Compute the covariate from ground elevation, normalized and centered at zero.

    Parameters:
        elev (np.ndarray): Array of elevation values (in meters).

    Returns:
        np.ndarray: Normalized and centered covariate values based on elevation.
    """
    # Normalize by the overall median elevation in Volume 12 (approximately 1500 meters)
    cov = elev / 1500.0
    return cov - 1.0  # Center covariate at zero


def load_cov2_data(config: dict, df_ams: pd.DataFrame) -> pd.DataFrame:
    """
    Populate the cov2 column in the AMS data based on the selected covariate option.

    Parameters:
        config (dict): Configuration dictionary containing data paths and plotting
        options.
        df_ams (pd.DataFrame): AMS data DataFrame.

    Returns:
        pd.DataFrame: AMS data with the cov2 column populated.
    """
    if config["ncov2_option"] == 0:  # Cov2 from log of co2
        df_ams = handle_co2_covariate(config, df_ams)
    elif config["ncov2_option"] == 1:  # Cov2 from GWL
        df_ams = handle_gwl_covariate(config, df_ams)
    elif config["ncov2_option"] == 2:  # Cov2 from year
        df_ams = df_ams.reset_index().sort_values(by=["id_num", "year"])
        df_ams["cov2"] = get_cov_from_year(df_ams["year"].values)
    else:
        df_ams = df_ams.reset_index().sort_values(by=["id_num", "year"])
        df_ams["cov2"] = np.nan  # Empty cov2
    return df_ams


def handle_co2_covariate(config: dict, df_ams: pd.DataFrame) -> pd.DataFrame:
    """
    Handle the co2-based covariate and add it to the AMS data.

    Parameters:
        config (dict): Configuration dictionary.
        df_ams (pd.DataFrame): AMS data DataFrame.

    Returns:
        pd.DataFrame: Updated AMS data with CO2-based covariate.
    """
    df_co2 = pd.read_table(
        Path(config["dir_data0"], "cmip6_CO2.txt"),
        sep="\t",
        dtype={"year": np.int16, "CO2": np.float32},
    )
    df_co2 = extrapolate_co2(df_co2)

    if config["save_plots"]:
        plot_co2_data(df_co2, config)

    df_ams = (
        df_ams.reset_index()
        .merge(df_co2, how="inner", on="year")
        .sort_values(by=["id_num", "year"])
    )
    df_ams["cov2"] = get_cov_from_co2(df_ams["CO2"].values)
    return df_ams


def extrapolate_co2(df_co2: pd.DataFrame) -> pd.DataFrame:
    """
    Extrapolate CO2 data to future years if necessary.

    Parameters:
        df_co2 (pd.DataFrame): CO2 data DataFrame.

    Returns:
        pd.DataFrame: CO2 data with extrapolated values.
    """
    n_co2 = len(df_co2.index)
    d_co2 = df_co2["CO2"].values[n_co2 - 1] - df_co2["CO2"].values[n_co2 - 2]
    for y in range(df_co2["year"].values[n_co2 - 1] + 1, 2030):
        df_co2.loc[n_co2] = [y, df_co2["CO2"].values[n_co2 - 1] + d_co2]
        n_co2 = len(df_co2.index)
    df_co2["CO2"] = df_co2["CO2"].round(2)
    return df_co2


def plot_co2_data(df_co2: pd.DataFrame, config: dict):
    """Plot and save CO2 data."""
    fig, ax = plt.subplots()
    ax.plot(df_co2["year"], df_co2["CO2"])
    ax.grid(which="major", linewidth=0.5, linestyle="dashed", color="gray")
    plt.xlabel("Year")
    plt.ylabel("CO2 (ppm)")
    plt.title("CMIP6 (roughly) global CO2 concentrations")
    fig.savefig(
        Path(config["proposedOutputPath"], "CO2.png"), bbox_inches="tight", dpi=300
    )
    plt.close(fig)


def handle_gwl_covariate(config: dict, df_ams: pd.DataFrame) -> pd.DataFrame:
    """
    Handle the GWL-based covariate and add it to the AMS data.

    Parameters:
        config (dict): Configuration dictionary.
        df_ams (pd.DataFrame): AMS data DataFrame.

    Returns:
        pd.DataFrame: Updated AMS data with GWL-based covariate.
    """
    df_gwl = pd.read_csv(Path(config["dir_data0"], "GWL_NOAA.csv"), index_col=False)
    df_gwl = extrapolate_gwl(df_gwl)

    if config["smooth_gwl"]:
        df_gwl = smooth_gwl(df_gwl)

    if config["save_plots"]:
        plot_gwl_data(df_gwl, config)

    df_ams = (
        df_ams.reset_index()
        .merge(df_gwl, how="inner", on="year")
        .sort_values(by=["id_num", "year"])
    )
    df_ams["cov2"] = (
        df_ams["GWLs"].values if config["smooth_gwl"] else df_ams["GWL"].values
    )
    return df_ams


def extrapolate_gwl(df_gwl: pd.DataFrame) -> pd.DataFrame:
    """Extrapolate GWL data to future years using linear fit."""
    n_gwl = len(df_gwl.index)
    df_gwl_70 = df_gwl[df_gwl["year"] >= 1970].copy()
    slope, intercept = np.polyfit(
        df_gwl_70["year"].values, df_gwl_70["GWL"].values, deg=1
    )
    for y in range(df_gwl["year"].values[n_gwl - 1] + 1, 2030):
        df_gwl.loc[n_gwl] = [y, intercept + slope * y]
        n_gwl = len(df_gwl.index)
    df_gwl["GWL"] = df_gwl["GWL"].round(3)
    return df_gwl


def smooth_gwl(df_gwl: pd.DataFrame) -> pd.DataFrame:
    """Smooth GWL data using a linear fit over a rolling window."""
    dk = 15
    df_gwl["GWLs"] = 0.0
    k0 = 0
    for k in range(dk, len(df_gwl.index) - dk - 1):
        slope, intercept = np.polyfit(
            df_gwl["year"].values[k - dk : k + dk + 1],
            df_gwl["GWL"].values[k - dk : k + dk + 1],
            deg=1,
        )
        df_gwl["GWLs"].values[k0 + 1 : k + 1] = (
            intercept + slope * df_gwl["year"].values[k0 + 1 : k + 1]
        )
        k0 = k
    df_gwl["GWLs"].values[k0 : len(df_gwl.index)] = (
        intercept + slope * df_gwl["year"].values[k0 : len(df_gwl.index)]
    )
    return df_gwl


def plot_gwl_data(df_gwl: pd.DataFrame, config: dict):
    """Plot and save GWL data and scatter plot comparing observed and smoothed GWL."""
    # Plot GWL historical
    fig, ax = plt.subplots()
    ax.scatter(
        df_gwl["year"], df_gwl["GWL"], label="observed", facecolors="none", edgecolors="k"
    )
    if config["smooth_gwl"]:
        ax.plot(df_gwl["year"], df_gwl["GWLs"], label="smoothed", c="r")
        plt.legend()
    ax.grid(which="major", linewidth=0.5, linestyle="dashed", color="gray")
    plt.xlabel("Year")
    plt.ylabel("GWL (deg C)")
    plt.title("Historical Global Warming Level by NOAA")
    fig.savefig(
        Path(config["proposedOutputPath"], "GWL.png"), bbox_inches="tight", dpi=300
    )
    plt.close(fig)

    # Plot GWL scatter
    fig, ax = plt.subplots(figsize=[8, 6])
    ax.scatter(
        df_gwl["GWL"],
        df_gwl["GWLs"],
        label="observed vs smoothed",
        facecolors="none",
        edgecolors="k",
    )
    ax.plot([-0.2, 1.2], [-0.2, 1.2], label="expected", c="r")
    plt.legend()
    ax.grid(which="major", linewidth=0.5, linestyle="dashed", color="gray")
    plt.xlabel("GWL observed (deg C)")
    plt.ylabel("GWL smoothed (deg C)")
    plt.title("Smoothing Historical Global Warming Level by NOAA")
    fig.savefig(
        Path(config["proposedOutputPath"], "GWL_scatter.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def read_elevation(config: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read elevation data from netCDF file."""
    ds_elev = nc.Dataset(
        Path(
            config["dir_data1"],
            f"elevation_SRTM180m_{config['region'].replace('V', 'Vol')}.nc",
        )
    )

    step = 2
    lats = ds_elev["lat"][::step]
    lons = ds_elev["lon"][::step]
    z = ds_elev["elevation"][::step, ::step]

    return lons, lats, z


def make_elevation_map(
    config: dict,
    sta_lon: np.ndarray,
    sta_lat: np.ndarray,
    sta_marker_size: float,
    sta_label: str,
    show_stations: bool,
) -> None:
    """Create and save elevation map."""
    lons, lats, z = read_elevation(config)
    xlon, ylat = np.meshgrid(lons, lats)

    fig = plt.figure(figsize=[10, 6])
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(config["mapExtent"])

    zmin, zmax = 0, 4000
    cbar_ticks = np.arange(zmin, zmax + 500, 500)
    crange = np.arange(zmin, zmax + 100, 100)

    f = ax.contourf(
        xlon,
        ylat,
        z,
        crange,
        transform=ccrs.PlateCarree(),
        cmap="terrain",
        vmin=zmin,
        vmax=zmax,
    )
    fig.colorbar(f, location="right", fraction=0.025, ticks=cbar_ticks)

    if show_stations:
        plt.title(f"Ground elevation (meters) with {sta_label} stations")
        ax.add_feature(cfeature.STATES, linewidths=0.1)
        plt.scatter(
            sta_lon,
            sta_lat,
            s=sta_marker_size,
            facecolors="none",
            edgecolors="k",
            linewidths=0.25,
        )

        fig_path = Path(
            config["proposedOutputPath"],
            config["ams_duration"],
            f"GroundElev_with_{sta_label}_Sta.png",
        )
        dpi = 500
    else:
        plt.title("Ground elevation (meters)")
        ax.add_feature(cfeature.STATES, linewidths=0.3)
        fig_path = Path(config["proposedOutputPath"], "GroundElev.png")
        dpi = 300

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def load_covariate_options(
    config: dict, df_meta: pd.DataFrame, df_ams: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Populate cov2 column and merge additional data into df_ams
    df_ams = load_cov2_data(config, df_ams)
    # Add normalized covariates to df_ams
    if config["ncov1_option"] == 0:  # MAP
        if "cov1" not in df_meta.columns:
            df_meta["cov1"] = get_cov_from_map(
                df_meta["prismMAP"], config["n_map_option"]
            )
        df_ams["cov1"] = get_cov_from_map(df_ams["prismMAP"], config["n_map_option"])
    elif config["ncov1_option"] == 1:  # MAM
        if "cov1" not in df_meta.columns:
            df_meta["cov1"] = get_cov_from_mam(df_meta["usedMAM"])
        df_ams["cov1"] = get_cov_from_mam(df_ams["usedMAM"])
    elif config["ncov1_option"] == 2:  # Elev
        if "cov1" not in df_meta.columns:
            df_meta["cov1"] = get_cov_from_elev(df_meta["elev_DEM"])
        df_ams["cov1"] = get_cov_from_elev(df_ams["elev_DEM"])

    if config["save_plots"]:
        make_elevation_map(
            config,
            df_meta.LON.values,
            df_meta.LAT.values,
            df_meta["nAMS"].values / 10,
            config["ams_duration"],
            True,
        )

    return df_meta, df_ams
