#!/usr/bin/env python3

## obs type class / configurations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
import os 
import yaml
import numpy as np

# Constants
# "da"             -> {exp}/CyclingDA/{cycle}/dbOut/obsout_da_{obs}.h5 (OMB + OMA)
# "hofx"           -> {exp}/{cycle}/dbOut/obsout_hofx_{obs}.h5 (OMB only)
# "hofx-verify-bg" -> {exp}/Verification/bg/mean/{cycle}/dbOut/obsout_hofx_{obs}.h5 (OMB only)
# "hofx-verify-bg-an" -> OMB from Verification/bg/mean as above, plus
#                        OMA from {exp}/Verification/an/mean/{cycle}/dbOut/obsout_hofx_{obs}.h5;
#                        common obs are defined separately (OMB across all bg, OMA across all an)
VALID_MODES = ("da", "hofx", "hofx-verify-bg", "hofx-verify-bg-an")

def has_oma(mode: str) -> bool:
    """Modes that carry an analysis (OMA)"""
    return mode in ("da", "hofx-verify-bg-an")

def get_verify_settings(config: Dict, iexp: int) -> Dict:
    """Per-experiment Verification sub-directory settings for the hofx-verify-* modes:
         Verification/{bg_dir|an_dir}/mean/{cycle}[/{lead_hr}hr]/dbOut
       Optional lists under `experiments` (one entry per experiment):
         verify_bg_dir  (default "bg"), verify_an_dir (default "an"),
         verify_lead_hr (default null -> no {lead_hr}hr directory)
    """
    exps = config["experiments"]
    num_exp = len(exps["names"])
    settings = {}
    for key, default in (("verify_bg_dir", "bg"), ("verify_an_dir", "an"), ("verify_lead_hr", None)):
        values = exps.get(key)
        if values is None:
            settings[key] = default
            continue
        if len(values) != num_exp:
            raise ValueError(f"experiments.{key} has {len(values)} entries, "
                             f"but experiments.names has {num_exp}")
        settings[key] = default if values[iexp] is None else values[iexp]
    return settings

def is_hofx_output(mode: str) -> bool:
    """Modes whose files come from the HofX application (hofx/EffectiveQC groups,
    no outer-loop iteration suffix)"""
    return mode in ("hofx", "hofx-verify-bg", "hofx-verify-bg-an")

DEFAULT_CONFIG = {
    "paths": {
        "exp_dir": "/glade/u/home/taosun/scratch/regional_mpasjedi",
        "output_dir": "/glade/u/home/taosun/scratch/regional_mpasjedi/verify_obs"
    },
    # one of VALID_MODES (see above)
    # Applies to every experiment in this run -- mixing da/hofx experiments in one
    # run is no longer supported (it never was in practice: all real configs so far
    # used the same type for every experiment).
    "mode": "da",
    "experiments": {
        "names": [
            "taosun_3dhybrid_3kmSG_ConvDA-new",
            "taosun_3dhybrid_3kmSG_ConvDA-V8.2.2",
        ],
        "labels": ["NSSL-branch", "Release"],
        "outerloop": [2, 2],
        "n_member": [1, 1],
    },
    "sensors": [
        "sondes", "profiler", "aircraft", "satwind", "satwnd", "sfc", "gnssrorefncep",
        "amsua_n18", "amsua_n19", "amsua_metop-b", "amsua_metop-c",
        "mhs_n19", "mhs_metop-b", "mhs_metop-c",
    ],
    "domains": {
        "global": [None, None, None, None],  # No masking
        "NXTro": [30.0, 90.0, -180.0, 180.0],
        "SXTro": [-90.0, -30.0, -180.0, 180.0],
        "NTro": [5.0, 30.0, -180.0, 180.0],
        "STro": [-30.0, -5.0, -180.0, 180.0],
        "ITCZ": [-5.0, 5.0, -180.0, 180.0],
        "Tro": [-23.5,  23.5, -180.0, 180.0]
    },
    "plot_settings": {
        "plot_oma": True,
        "apply_domains": ["global"],
        "save_graphics": True,
    },
    "stats_settings": {
        "ci_method": "bootstrap",
        "qc_flag": [0],
        "check_obsvalue": True,
    },
    "time_settings": {
        "start_time": "2024-09-13 06:00:00",
        "end_time": "2024-09-21 12:00:00",
        "interval": 6,
    }
}


@dataclass
class ObservationConfig:
    """Configuration for observation types and their properties"""
    variables: List[Union[str, int]]
    units: List[str]
    labels: List[str]
    tags: List[str]


OBSERVATION_CONFIGS = {
    "sondes": ObservationConfig(
        variables=['windNorthward', 'windEastward', 'airTemperature', 'specificHumidity'],
        tags=['U','V','T','Q'],
        units=[r'$\mathrm{m \cdot s^{-1}}$', r'$\mathrm{m \cdot s^{-1}}$', r'$\mathrm{K}$', r'$\mathrm{g \cdot kg^{-1}}$'],
        labels=['(a) U', '(b) V', '(c) T', '(d) Q']
    ),
    "aircraft": ObservationConfig(
        variables=['windNorthward', 'windEastward', 'airTemperature', 'specificHumidity'],
        tags=['U','V','T','Q'],
        units=[r'$\mathrm{m \cdot s^{-1}}$', r'$\mathrm{m \cdot s^{-1}}$', r'$\mathrm{K}$', r'$\mathrm{g \cdot kg^{-1}}$'],
        labels=['(a) U', '(b) V', '(c) T', '(d) Q']
    ),
    "satwind": ObservationConfig(
        variables=['windNorthward', 'windEastward'],
        tags=['U','V'],
        units=[r'$\mathrm{m \cdot s^{-1}}$', r'$\mathrm{m \cdot s^{-1}}$'],
        labels=['(a) U', '(b) V']
    ),
    "satwnd": ObservationConfig(
        variables=['windNorthward', 'windEastward'],
        tags=['U','V'],
        units=[r'$\mathrm{m \cdot s^{-1}}$', r'$\mathrm{m \cdot s^{-1}}$'],
        labels=['(a) U', '(b) V']
    ),
    "profiler": ObservationConfig(
        variables=['windNorthward', 'windEastward'],
        tags=['U','V'],
        units=[r'$\mathrm{m \cdot s^{-1}}$', r'$\mathrm{m \cdot s^{-1}}$'],
        labels=['(a) U', '(b) V']
    ),
    "sfc": ObservationConfig(
        variables=['windNorthward', 'windEastward', 'airTemperature', 'specificHumidity', 'stationPressure'],
        tags=['U10','V10','T2','Q2','Ps'],
        units=[r'$\mathrm{m \cdot s^{-1}}$', r'$\mathrm{m \cdot s^{-1}}$', r'$\mathrm{K}$', r'$\mathrm{g \cdot kg^{-1}}$', 'Pa'],
        labels=['(a) U10', '(b) V10', '(c) T2', '(d) Q2', '(e) Ps']
    ),
    "gnssrobndropp1d*": ObservationConfig(
        variables=['bendingAngle'],
        tags=['bend_ang'],
        units=['%'],
        labels=['Bending angle']
    ),
    "gnssrorefncep*": ObservationConfig(
        variables=['atmosphericRefractivity'],
        tags=['refract'],
        units=['%'],
        labels=['Refractivity']
    ),
    "radar_rw": ObservationConfig(
        variables=['radialVelocity'],
        tags=['rad_vel'],
        units=[r'$\mathrm{m \cdot s^{-1}}$'],
        labels=['Radial velocity']
    ),
    "radar_zh*": ObservationConfig(
        variables=['equivalentReflectivityFactor'],
        tags=['reflect'],
        units=['dBZ'],
        labels=['Reflectivity']
    ),
    "amsua_aqua": ObservationConfig(
        variables=[8, 9],
        tags=['ch8','ch9'],
        units=['K', 'K'],
        labels=['(a) Channel 8', '(b) Channel 9']
    ),
    "amsua_metop-a": ObservationConfig(
        variables=[5, 6, 9],
        tags=['ch5','ch6','ch9'],
        units=['K', 'K', 'K'],
        labels=['(a) Channel 5', '(b) Channel 6', '(c) Channel 9']
    ),
    "amsua_metop-b": ObservationConfig(
        variables=[8, 9],
        tags=['ch8','ch9'],
        units=['K', 'K', 'K'],
        labels=['(a) Channel 8', '(b) Channel 9']
    ),
    "amsua_metop-c": ObservationConfig(
        variables=[4, 5, 6, 7, 8, 9, 10, 11],
        tags=['ch4','ch5','ch6','ch7','ch8','ch9','ch10','ch11'],
        units=['K', 'K','K', 'K', 'K', 'K', 'K', 'K'],
        labels=['(a) Channel 4', '(b) Channel 5', '(c) Channel 6', '(d) Channel 7',
                '(e) Channel 8', '(f) Channel 9', '(g) Channel 10', '(h) Channel 11']
    ),
    "amsua_n15": ObservationConfig(
        variables=[5, 7, 8, 9],
        tags=['ch5','ch7','ch8','ch9'],
        units=['K', 'K', 'K' , 'K'],
        labels=['(a) Channel 5', '(b) Channel 7', '(c) Channel 8', '(d) Channel 9']
    ),
    "amsua_n18": ObservationConfig(
        variables=[5, 6, 7, 8, 9],
        tags=['ch5','ch6','ch7','ch8','ch9'],
        units=['K', 'K', 'K', 'K', 'K'],
        labels=['(a) Channel 5', '(b) Channel 6', '(c) Channel 7', '(d) Channel 8', '(e) Channel 9']
    ),
    "amsua_n19": ObservationConfig(
        variables=[5, 6, 7, 9],
        tags=['ch5','ch6','ch7','ch9'],
        units=['K', 'K', 'K', 'K'],
        labels=['(a) Channel 5', '(b) Channel 6', '(c) Channel 7', '(d) Channel 9']
    ),
    "mhs_n19": ObservationConfig(
        variables=[4, 5],
        tags=['ch4','ch5'],
        units=['K', 'K'],
        labels=['(a) Channel 4', '(b) Channel 5']
    ),
    "mhs_n18": ObservationConfig(
        variables=[3, 4, 5],
        tags=['ch3','ch4','ch5'],
        units=['K', 'K', 'K'],
        labels=['(a) Channel 3', '(b) Channel 4', '(c) Channel 5']
    ),
    "mhs_metop-a": ObservationConfig(
        variables=[3, 4, 5],
        tags=['ch3','ch4','ch5'],
        units=['K', 'K', 'K'],
        labels=['(a) Channel 3', '(b) Channel 4', '(c) Channel 5']
    ),
    "mhs_metop-b": ObservationConfig(
        variables=[3, 4, 5],
        tags=['ch3','ch4','ch5'],
        units=['K', 'K', 'K'],
        labels=['(a) Channel 3', '(b) Channel 4', '(c) Channel 5']
    ),
    "mhs_metop-c": ObservationConfig(
        variables=[3, 4, 5],
        tags=['ch3','ch4','ch5'],
        units=['K', 'K', 'K'],
        labels=['(a) Channel 3', '(b) Channel 4', '(c) Channel 5']
    ),
    "amsua-cld": ObservationConfig(
        variables=[1, 2, 3, 4, 15],
        tags=['ch1','ch2','ch3','ch4','ch15'],
        units=['K', 'K', 'K', 'K', 'K'],
        labels=['(a) Channel 1', '(b) Channel 2', '(c) Channel 3', '(d) Channel 4', '(e) Channel 15']
    ),
    "ahi-cld_himawari9": ObservationConfig(
        variables=[2, 3, 4, 7],
        tags=['ch8','ch9','ch10','ch13'],
        units=['K', 'K', 'K', 'K', 'K'],
        labels=['(a) Channel 8', '(b) Channel 9', '(c) Channel 10', '(d) Channel 13']
    )
}

def load_config(config_path: Optional[str] = None) -> Dict:
    """Load configuration from file or use defaults"""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return DEFAULT_CONFIG

def get_obs_config(diag_obs: str) -> ObservationConfig:
    # exact match first
    if diag_obs in OBSERVATION_CONFIGS:
        return OBSERVATION_CONFIGS[diag_obs]

    # wildcard match
    for key, cfg in OBSERVATION_CONFIGS.items():
        if key.endswith('*') and diag_obs.startswith(key[:-1]):
            return cfg

    raise ValueError(f"No ObservationConfig found for {diag_obs}")

def get_bin_method(diag_obs: str) -> int:
    """Determine binning method based on observation type"""
    if diag_obs.startswith(("amsua", "mhs", "ahi", "abi", "iasi", "atms")):
        return 3  # radiances
    if diag_obs.startswith("gnssro") or diag_obs.startswith("radar"):
        return 2  # height
    return 1  # pressure
    
def get_levels(bin_method: int, bin_width=None, bin_bottom=None, bin_top=None) -> np.ndarray:
    """Get vertical levels based on binning method"""
    if bin_width and bin_bottom and bin_top:
        return np.arange(bin_bottom, bin_top+bin_width/2., bin_width)
    if bin_method == 1:  # pressure
        return np.array([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50])
    if bin_method == 2:  # height
        interval = 2.0
        return np.arange(interval/2., 15*interval+1.0, interval)
    if bin_method == 3:  # latitude bins for radiances
        interval = 10.0
        return np.arange(-80.0, 80.0+interval, interval)
        #return np.arange(-10.0, 10.0+interval, interval)