import logging
import math
import time
from pathlib import Path
from typing import Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} execution time: {end_time - start_time:.6f} seconds")
        return result

    return wrapper


def compute_nams(
    df_meta: pd.DataFrame, df_ams: pd.DataFrame, config: dict, sdur: str
) -> pd.DataFrame:
    """Compute and add the nAMS column to the metadata DataFrame."""
    df_meta = df_meta.merge(df_ams["id_num"].value_counts(), how="inner", on="id_num")
    df_meta = df_meta.rename(columns={"count": "nAMS"})
    if config["save_plots"]:
        plot_nams_map(df_meta, config, sdur)
    return df_meta


def compute_mean_ams(
    df_meta: pd.DataFrame, df_ams: pd.DataFrame, config: dict, sdur: str
) -> pd.DataFrame:
    """Compute and add the meanAMS column to the metadata DataFrame."""
    df_meta = df_meta.merge(
        df_ams.groupby("id_num")["precip"].mean(), how="left", on="id_num"
    )
    df_meta["precip"] = df_meta["precip"].round(4)
    df_meta = df_meta.rename(columns={"precip": "meanAMS"})
    if config["save_plots"]:
        plot_mean_ams_map(df_meta, config, sdur)
    df_meta["gridMAM"] = df_meta["gridMAM"].fillna(df_meta["meanAMS"]).round(4)
    return df_meta


def compute_used_mam(df_meta: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compute and add the usedMAM column to the metadata DataFrame."""
    if config["use_station_mam"]:
        df_meta["usedMAM"] = df_meta["meanAMS"].where(
            df_meta["nAMS"] > config["ams_nmax_station_mam"], df_meta["gridMAM"]
        )
        df_meta["usedMAM"] = (
            df_meta["usedMAM"]
            .where(
                (df_meta["nAMS"] < config["ams_nmin_station_mam"])
                | (config["ams_nmax_station_mam"] < df_meta["nAMS"]),
                (df_meta["meanAMS"] - df_meta["gridMAM"])
                * (df_meta["nAMS"] - config["ams_nmin_station_mam"])
                / (config["ams_nmax_station_mam"] - config["ams_nmin_station_mam"])
                + df_meta["gridMAM"],
            )
            .round(4)
        )
    else:
        df_meta["usedMAM"] = df_meta["gridMAM"]
    return df_meta


def plot_nams_map(df_meta: pd.DataFrame, config: dict, sdur: str):
    """Plot the map of nAMS values at stations."""
    fig = plt.figure(figsize=[10, 6])
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.set_extent(
        [
            math.floor(df_meta.LON.min()) - 1,
            math.ceil(df_meta.LON.max() + 1),
            math.floor(df_meta.LAT.min()) - 1,
            math.ceil(df_meta.LAT.max() + 1),
        ]
    )
    ax.add_feature(cfeature.STATES, linewidths=0.3)
    points = ax.scatter(
        df_meta.LON,
        df_meta.LAT,
        transform=ccrs.PlateCarree(),
        c=df_meta["nAMS"],
        s=7,
        alpha=0.6,
        cmap="jet",
    )
    fig.colorbar(points)
    plt.title(f"Number of {sdur} Annual Maximum Values at Stations")
    fig.savefig(
        Path(
            config["proposedOutputPath"],
            config["ams_duration"],
            f"Station_nAMS_{sdur}.png",
        ),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def plot_mean_ams_map(df_meta: pd.DataFrame, config: dict, sdur: str):
    """Plot the map of meanAMS values at stations."""
    fig = plt.figure(figsize=[10, 6])
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.set_extent(
        [
            math.floor(df_meta.LON.min()) - 1,
            math.ceil(df_meta.LON.max() + 1),
            math.floor(df_meta.LAT.min()) - 1,
            math.ceil(df_meta.LAT.max() + 1),
        ]
    )
    ax.add_feature(cfeature.STATES, linewidths=0.3)
    points = ax.scatter(
        df_meta.LON,
        df_meta.LAT,
        transform=ccrs.PlateCarree(),
        c=df_meta["meanAMS"],
        s=7,
        alpha=0.6,
        cmap="jet",
    )
    fig.colorbar(points)
    plt.title(f"Mean Annual Maximum (MAM_{sdur}) Precipitation at Stations")
    fig.savefig(
        Path(
            config["proposedOutputPath"],
            config["ams_duration"],
            f"Station_MAM_{sdur}.png",
        ),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


# @timing_decorator
def read_ams(config: dict, sdur: str, df_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Load the annual maximum series for the entire Volume 12 project at a given duration.

    Parameters:
        config (dict): Configuration dictionary containing directory and plotting options.
        sdur (str): Duration for which the AMS data is loaded.
        df_meta (pd.DataFrame): Metadata DataFrame for the stations.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Updated AMS data and metadata DataFrames.
    """
    # Load the AMS data from a CSV file
    df_ams = pd.read_csv(Path(config["dir_data1"], f"df_ams_{sdur}.csv")).set_index(
        "id_num"
    )
    df_ams = df_ams.reset_index()
    n0 = len(df_ams.index)

    # Filter AMS data by year range if specified
    if config["ams_year_range"][0] > 0:
        df_ams = df_ams[df_ams["year"] >= config["ams_year_range"][0]]
    if config["ams_year_range"][1] > 0:
        df_ams = df_ams[df_ams["year"] <= config["ams_year_range"][1]]

    if (n0 > len(df_ams.index)) and config["save_plots"]:
        print(
            "Number of AMS values removed outside the year range "
            f"[{config['ams_year_range'][0]}, {config['ams_year_range'][1]}] "
            f"= {n0 - len(df_ams.index)} out of {n0}"
        )

    if config["okUseCovMAP"]:
        nams0 = len(df_ams.index)
        df_ams = df_ams[df_ams["id_num"].isin(df_meta.index)]
        if config["save_plots"]:
            print(
                f"Number of removed {sdur} AMS values with prismMAP = NaN: "
                f"{nams0 - len(df_ams.index)}"
            )

    return df_ams


# @timing_decorator
def load_stats_to_meta(
    config: dict, df_meta: pd.DataFrame, df_ams: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Merge AMS data with metadata to add nAMS, meanAMS, and other computed columns
    if "nAMS" not in df_meta.columns:
        df_meta = compute_nams(df_meta, df_ams, config, config["ams_duration"])

    if "meanAMS" not in df_meta.columns:
        df_meta = compute_mean_ams(df_meta, df_ams, config, config["ams_duration"])

    if "usedMAM" not in df_meta.columns:
        df_meta = compute_used_mam(df_meta, config)

    if "start_year" not in df_meta.columns:
        df_meta = df_meta.merge(
            df_ams.groupby("id_num")["year"].min(), how="inner", on="id_num"
        )
        df_meta = df_meta.rename(columns={"year": "start_year"})

    if "end_year" not in df_meta.columns:
        df_meta = df_meta.merge(
            df_ams.groupby("id_num")["year"].max(), how="inner", on="id_num"
        )
        df_meta = df_meta.rename(columns={"year": "end_year"})

    df_ams = df_ams.merge(df_meta[["meanAMS"]], how="left", on="id_num")
    df_ams = df_ams.merge(
        df_meta[["gridMAM", "usedMAM", "prismMAP", "elev_DEM"]], how="left", on="id_num"
    )

    return df_meta, df_ams
