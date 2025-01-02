from typing import Dict, List

from pydantic import (
    BaseModel,
    StrictBool,
    StrictInt,
    StrictStr,
    confloat,
    conint,
    conlist,
)


class RegionSpecificConfig(BaseModel):
    sdurl: List[str]
    diff_map0: List[float]
    diff_elev0: List[int]
    elev_range0: List[int]
    obstacle_height0: List[int]
    diff_mam0: List[conlist(float)]  # List of lists
    p2st0: List[conlist(float)]  # List of lists


class MapExtents(BaseModel):
    V11: conlist(confloat(ge=-180.0, le=180.0), min_length=4, max_length=4)
    V12: conlist(confloat(ge=-180.0, le=180.0), min_length=4, max_length=4)
    V12_MT: conlist(confloat(ge=-180.0, le=180.0), min_length=4, max_length=4)
    V13: conlist(confloat(ge=-180.0, le=180.0), min_length=4, max_length=4)
    CONUS: conlist(confloat(ge=-180.0, le=180.0), min_length=4, max_length=4)


class DirData(BaseModel):
    V11: StrictStr
    V12: StrictStr
    V13: StrictStr
    CONUS: StrictStr


class ConfigValidate(BaseModel):
    region: StrictStr
    dir_data0: StrictStr
    prepared_data_region: DirData
    prism_file: StrictStr
    meta_file: StrictStr
    dist2coast_file: StrictStr
    fit_kappa: StrictBool
    solve_at_station: StrictBool
    remove_center_station: StrictBool
    ams_year_range: conlist(int, min_length=2, max_length=2)
    use_saved_24h_weights: StrictBool
    recompute_weights: StrictBool
    arithmetic_mean_weights: StrictBool
    use_saved_weights: StrictBool
    ams_duration: StrictStr
    min_radius: conint(gt=0)
    max_radius: conint(gt=0)
    min_diff_mam: conint(gt=0)
    max_diff_mam: conint(gt=0)
    min_diff_map: conint(gt=0)
    max_diff_map: conint(gt=0)
    min_diff_elev: conint(gt=0)
    max_diff_elev: conint(gt=0)
    min_elev_range: conint(gt=0)
    max_elev_range: conint(gt=0)
    min_obstacle_height: conint(gt=0)
    max_obstacle_height: conint(gt=0)
    min_log10p2st: confloat(ge=0)
    max_log10p2st: confloat(ge=0)
    min_diff_dist2coast: confloat(gt=0)
    max_diff_dist2coast: confloat(gt=0)
    use_weights_minmax_perc: StrictBool
    wmin_perc: conint(ge=0, le=100)
    wmax_perc: conint(ge=0, le=100)
    perc0: List[float]
    apply_cap2limits: StrictBool
    min0_diff_mam: conint(gt=0)
    min0_diff_map: conint(gt=0)
    min0_diff_elev: conint(gt=0)
    min0_elev_range: conint(gt=0)
    min0_obstacle_height: conint(gt=0)
    max0_p2st: confloat(ge=0, le=1)
    min0_p2st: confloat(ge=0, le=1)
    min0_diff_dist2coast: confloat(gt=0)
    max0_diff_dist2coast: confloat(gt=0)
    compute_obstacle_height: StrictBool
    obstacle_height_step: conint(ge=0)
    compute_2sample_tests: StrictBool
    ams_nmin_2sample_tests: conint(ge=0)
    use_hourly_diff_map_weights: StrictBool
    use_dist2coast: StrictBool
    dist2coast0: confloat(gt=0)
    nmodel: StrictInt
    relax_bounds: StrictBool
    bounds_percentile: confloat(ge=0.5, le=1.0)
    ncov2_option: StrictInt
    smooth_gwl: StrictBool
    ncov2term_option: StrictInt
    ncov1_option: StrictInt
    n_map_option: StrictInt
    use_station_mam: StrictBool
    use_greg_mam: StrictBool
    ams_nmin_station_mam: conint(ge=0)
    ams_nmax_station_mam: conint(ge=0)
    errors_in_inches: StrictBool
    return_period_out: List[float]
    out_xobs: StrictBool
    output_a14_precip: StrictBool
    save_plots: StrictBool
    ci_perc: confloat(ge=0.0, le=1.0)
    ci_niter: conint(gt=0)
    ci_okrho: StrictBool
    ci_std_factor: confloat(gt=0.0)
    min_ci_std_factor: confloat(ge=0.0)
    max_ci_std_factor: confloat(ge=0.0)
    ci_r_vs_d_nmin_station: conint(ge=0)
    ci_r_vs_d_weight_min: confloat(ge=0.0)
    ci_rmin: confloat(ge=0.0)
    pp_fdur0: confloat(gt=0.0)
    pp_fdur1: confloat(gt=0.0)
    region_specific: Dict[str, RegionSpecificConfig]
    prepared_data_prefix: str
    pre_industrial_co2: float
    current_co2: float
    pre_industrial_gwl: float
    current_gwl: float
    base_year: int
    current_year: int
    map_extents: MapExtents
    weights_log_file: StrictStr
    fit_log_file: StrictStr
    log_level: StrictStr
    output_base_path: StrictStr
