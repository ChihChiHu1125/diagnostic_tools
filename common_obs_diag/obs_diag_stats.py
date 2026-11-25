#!/usr/bin/env python3

## subroutines for calculating stats
import os, sys, glob, traceback
import h5py
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union
from scipy import stats
import pandas as pd

# obs type config / class:
import obs_diag_config as odc

# Upper bound on a "sane" ObsError value used when reconstructing QC from
# ObsError for LETKF/GETKF output that has no EffectiveQC group.
MAX_VALID_OBSERROR = 10000.0

# GNSSRO variables whose innovation is reported relative to the obs, (O-B)/O*100 [%]
GNSSRO_RELATIVE_VARS = ("atmosphericRefractivity", "bendingAngle")

# h5 paths already reported missing by build_filename (warn once per file)
_warned_missing_files = set()


def qc_from_obserror(obserror: np.ndarray, check_range: bool) -> np.ndarray:
    """Reconstruct a pass/fail QC flag from ObsError when EffectiveQC isn't
    written (current LETKF/GETKF output has no EffectiveQC group).
    Returns 0 where the observation passes, NaN where it fails -- consistent
    with how EffectiveQC-derived qc arrays are consumed elsewhere (only
    `qc == 0` is ever selected as "qualified").
    """
    if check_range:
        valid = (obserror > 0.) & (obserror <= MAX_VALID_OBSERROR)
    else:
        valid = ~np.isnan(obserror)
    return np.where(valid, 0, np.nan)


def combine_common_qc(common_qc: np.ndarray, qc_selected: np.ndarray,
                      ref_obsvalue: Optional[np.ndarray], obsvalue: Optional[np.ndarray],
                      check_obsvalue: bool, tag: str) -> np.ndarray:
    """AND one more experiment's QC selection into the running common_qc.
    With check_obsvalue, an obs only stays common if its ObsValue also matches
    the reference experiment's (same index -> same observation). Arrays of
    different length can't be matched by index, so nothing is common then.
    """
    if len(qc_selected) != len(common_qc):
        print(f"WARNING: {tag}: nlocs {len(qc_selected)} != reference {len(common_qc)}; "
              f"no common obs for this cycle", file=sys.stderr)
        return np.zeros_like(common_qc, dtype=bool)

    common_qc = common_qc & qc_selected

    if check_obsvalue:
        if ref_obsvalue is None or obsvalue is None:
            print(f"WARNING: {tag}: ObsValue missing, skipping ObsValue check", file=sys.stderr)
        else:
            same_obs = np.isclose(obsvalue, ref_obsvalue, rtol=1e-5, atol=0., equal_nan=False)
            n_diff = np.sum(common_qc & ~same_obs)
            if n_diff > 0:
                print(f"WARNING: {tag}: {n_diff} QC-passed obs have different ObsValue "
                      f"from the reference; excluded from common obs", file=sys.stderr)
            common_qc = common_qc & same_obs

    return common_qc


def build_filename(base_dir: str, exp_name: str, mode: str, exp_member: int,
                  cycle_time: datetime, diag_obs: str,
                  verify: Optional[Dict] = None) -> Dict[str, str]:
    """Construct the appropriate filename based on parameters.
       Falls back to obsout_da/hofx_{diag_obs}*.h5 if exact file doesn't exist.
       `mode` is the run-wide setting (config["mode"]), one of odc.VALID_MODES.
       `verify` is this experiment's odc.get_verify_settings() (hofx-verify-* modes).
    """
    date_str = cycle_time.strftime("%Y%m%d%H")
    filedict = {"bak": None, "ana": None}

    # --------------------------------------------------
    # Helper: resolve with wildcard fallback
    # --------------------------------------------------
    def resolve(fname):
        if fname is None:
            return None
        if os.path.exists(fname):
            return fname
        matches = glob.glob(fname.replace(".h5", "*.h5"))
        if not matches:
            # build_filename is called several times per file (per variable, per
            # pass), so only warn the first time a path is found missing
            if fname not in _warned_missing_files:
                _warned_missing_files.add(fname)
                print(f"WARNING: file not found, skipped: {fname}", file=sys.stderr)
            return None
        return sorted(matches)[-1]

    # --------------------------------------------------
    # PROFILER uses sondes file
    # --------------------------------------------------
    obs_name = "sondes" if diag_obs == "profiler" else diag_obs

    if mode == 'da':
        # ---- background (OMB / hofx0)
        bak = (
            f"{base_dir}/{exp_name}/CyclingDA/{date_str}/dbOut/"
            f"obsout_da_{obs_name}.h5"
        )
        filedict["bak"] = resolve(bak)

        # ---- analysis (OMA)
        if exp_member > 1:
            ana = (
                f"{base_dir}/{exp_name}/CyclingDA/{date_str}/dbOut/"
                f"obsout_da_{obs_name}_ana.h5"
            )
            filedict["ana"] = resolve(ana)
        else:
            # single-member DA: analysis == background
            filedict["ana"] = filedict["bak"]
    # ==================================================
    # HOFX VERIFICATION OF THE BACKGROUND (Verification/{bg_dir}, mean only)
    # (+ the analysis from Verification/{an_dir} in hofx-verify-bg-an)
    # ==================================================
    elif mode in ('hofx-verify-bg', 'hofx-verify-bg-an'):

        if verify is None:
            verify = {"verify_bg_dir": "bg", "verify_an_dir": "an", "verify_lead_hr": None}
        # e.g. Verification/fc/mean/{cycle}/0hr/dbOut for HofX of a state named "fc"
        lead_dir = "" if verify["verify_lead_hr"] is None else f"/{verify['verify_lead_hr']}hr"

        bak = (
            f"{base_dir}/{exp_name}/Verification/{verify['verify_bg_dir']}/mean/"
            f"{date_str}{lead_dir}/dbOut/obsout_hofx_{obs_name}.h5"
        )
        filedict["bak"] = resolve(bak)

        # ---- analysis (OMA) from HofX of the analysis, only in hofx-verify-bg-an
        if mode == 'hofx-verify-bg-an':
            ana = (
                f"{base_dir}/{exp_name}/Verification/{verify['verify_an_dir']}/mean/"
                f"{date_str}{lead_dir}/dbOut/obsout_hofx_{obs_name}.h5"
            )
            filedict["ana"] = resolve(ana)
        else:
            filedict["ana"] = None

    # ==================================================
    # HOFX ONLY
    # ==================================================
    else:

        bak = (
            f"{base_dir}/{exp_name}/{date_str}/dbOut/"
            f"obsout_hofx_{obs_name}.h5"
        )
        filedict["bak"] = resolve(bak)

        # no analysis for hofx-only experiments
        filedict["ana"] = None

    return filedict

def load_observation_data(filename: str, diag_obs: str, diag_var: Union[str, int],
                        bin_method: int, outerloop: int, nmember: int, only_qc: bool = False):
    """Load observation data from HDF5 file for OMB and OMA respectively"""
    # if only_qc = True, only read and load the qc data
    
    obs_vector = {
        'lat': None,
        'lon': None,
        'hgt': None,
        'pres': None,
        'innov': None,
        'innov_nobc': None,
        'qc': None,
        'spread': None,
        'totSpread': None,
        'obsvalue': None,
    }

    try:
        with h5py.File(filename, 'r') as f:

            # ObsValue (raw units) -- used to verify common obs are the same obs across experiments
            if 'ObsValue' in f:
                if bin_method == 3:  # radiances
                    obs_vector['obsvalue'] = f['ObsValue']['brightnessTemperature'][:, diag_var-1]
                elif diag_var in f['ObsValue']:
                    obs_vector['obsvalue'] = f['ObsValue'][diag_var][:]

            if (only_qc == False):
                # --------------------------------------------------
                # MetaData
                # --------------------------------------------------
                obs_vector['pres'] = f['MetaData']['pressure'][:] / 100. if 'pressure' in f['MetaData'] else None
                obs_vector['hgt']  = f['MetaData']['height'][:] / 1000. if 'height' in f['MetaData'] else None
                obs_vector['lat']  = f['MetaData']['latitude'][:]
                obs_vector['lon']  = f['MetaData']['longitude'][:]
                obs_vector['lon']  = np.where(obs_vector['lon'] > 180., obs_vector['lon']-360., obs_vector['lon'])

            # Data assimilation files have ombg/oman
            if bin_method == 3:  # radiances

                if (only_qc == False):
                    ## Innovation
                    if outerloop == 0:
                        if 'ombg' in f:
                            obs_vector['innov'] = f['ombg']['brightnessTemperature'][:, diag_var-1]
                            if 'ObsBias0' in f:
                                obs_vector['innov_nobc'] = f['ombg']['brightnessTemperature'][:, diag_var-1] + f['ObsBias0']['brightnessTemperature'][:, diag_var-1]
                        elif 'hofx' in f:
                            obs_vector['innov'] = f['ObsValue']['brightnessTemperature'][:, diag_var-1] - f['hofx']['brightnessTemperature'][:, diag_var-1]
                            if 'ObsBias' in f:
                                # NOTE: was `f['ombg'] + f['ObsBias']`, but this branch is only
                                # reached when 'ombg' is NOT in the file (see the `if 'ombg' in f`
                                # above) -- that raised a KeyError whenever ObsBias existed without
                                # ombg. Use the same ObsValue-(hofx-bias) form as the OMA branches.
                                obs_vector['innov_nobc'] = f['ObsValue']['brightnessTemperature'][:, diag_var-1] - (f['hofx']['brightnessTemperature'][:, diag_var-1] - f['ObsBias']['brightnessTemperature'][:, diag_var-1])
                        elif 'hofx0' in f:
                            obs_vector['innov'] = f['ObsValue']['brightnessTemperature'][:, diag_var-1] - f['hofx0']['brightnessTemperature'][:, diag_var-1]
                            if 'ObsBias0' in f:
                                # NOTE: same fix as the 'hofx' branch above -- 'ombg' isn't
                                # guaranteed to exist here either.
                                obs_vector['innov_nobc'] = f['ObsValue']['brightnessTemperature'][:, diag_var-1] - (f['hofx0']['brightnessTemperature'][:, diag_var-1] - f['ObsBias0']['brightnessTemperature'][:, diag_var-1])
                        elif 'hofx0_mean' in f:
                            obs_vector['innov'] = f['ObsValue']['brightnessTemperature'][:, diag_var-1] - f['hofx0_mean']['brightnessTemperature'][:, diag_var-1]
                    else:
                        if 'oman' in f:
                            obs_vector['innov'] = f['oman']['brightnessTemperature'][:, diag_var-1]
                            if f'ObsBias{outerloop}' in f:
                                obs_vector['innov_nobc'] = f['ObsValue']['brightnessTemperature'][:, diag_var-1] - (f[f'hofx{outerloop}']['brightnessTemperature'][:, diag_var-1] - f[f'ObsBias{outerloop}']['brightnessTemperature'][:, diag_var-1])
                        elif f'hofx{outerloop}' in f:
                            obs_vector['innov'] = f['ObsValue']['brightnessTemperature'][:, diag_var-1] - f[f'hofx{outerloop}']['brightnessTemperature'][:, diag_var-1]
                            if f'ObsBias{outerloop}' in f:
                                obs_vector['innov_nobc'] = f['ObsValue']['brightnessTemperature'][:, diag_var-1] - (f[f'hofx{outerloop}']['brightnessTemperature'][:, diag_var-1] - f[f'ObsBias{outerloop}']['brightnessTemperature'][:, diag_var-1])
                        elif 'hofx0_mean' in f:
                            obs_vector['innov'] = f['ObsValue']['brightnessTemperature'][:, diag_var-1] - f['hofx0_mean']['brightnessTemperature'][:, diag_var-1]
        
                ## quality control flags            
                if f'EffectiveQC{outerloop}' in f:
                    obs_vector['qc'] = f[f'EffectiveQC{outerloop}']['brightnessTemperature'][:, diag_var-1]
                elif (outerloop == 0) and ('EffectiveQC' in f): # hofx application output has no iteration suffix
                    obs_vector['qc'] = f['EffectiveQC']['brightnessTemperature'][:, diag_var-1]
                elif (nmember > 1) and ('ObsError' in f): #### this is for current LETKF/GETKF cuz not EffectiveQC is available.
                    obserror = f['ObsError']['brightnessTemperature'][:, diag_var-1]
                    obs_vector['qc'] = qc_from_obserror(obserror, check_range=False)
    
                ## spread
                if nmember > 1:
                    # Spread
                    hofx_ens = []
                    for iens in range(nmember):
                        hofx = f[f'hofx0_{iens+1}']['brightnessTemperature'][:, diag_var-1]
                        hofx_ens.append(hofx)
                    hofx_ens = np.stack(hofx_ens, axis=0)
                    obs_vector['spread'] = np.nanstd(hofx_ens, axis=0, ddof=0)
                    # total spread
                    obserr = f['ObsError']['brightnessTemperature'][:, diag_var-1]
                    obs_vector['totSpread'] = np.sqrt(np.nan_to_num(obs_vector['spread'], nan=0.)**2+
                        np.nan_to_num(obserr, nan=0.0)**2.0)
                    
            else: # non-radiance
                ## Innovations

                if (only_qc == False):
                    if outerloop == 0:
                        if 'ombg' in f:
                            obs_vector['innov'] = f['ombg'][diag_var][:]
                        elif 'hofx' in f:
                            obs_vector['innov'] = f['ObsValue'][diag_var][:] - f['hofx'][diag_var][:]
                        elif 'hofx0' in f:
                            obs_vector['innov'] = f['ObsValue'][diag_var][:] - f['hofx0'][diag_var][:]
                        elif 'hofx0_mean' in f:
                            obs_vector['innov'] = f['ObsValue'][diag_var][:] - f['hofx0_mean'][diag_var][:]
                    else:
                        if 'oman' in f:
                            obs_vector['innov'] = f['oman'][diag_var][:]
                        elif f'hofx{outerloop}' in f:
                            obs_vector['innov'] = f['ObsValue'][diag_var][:] - f[f'hofx{outerloop}'][diag_var][:]
                        elif 'hofx0_mean' in f:
                            obs_vector['innov'] = f['ObsValue'][diag_var][:] - f['hofx0_mean'][diag_var][:]

                ## quality control flags
                if f'EffectiveQC{outerloop}' in f:
                    obs_vector['qc'] = f[f'EffectiveQC{outerloop}'][diag_var][:]
                elif (outerloop == 0) and ('EffectiveQC' in f): # hofx application output has no iteration suffix
                    obs_vector['qc'] = f['EffectiveQC'][diag_var][:]
                elif (nmember > 1) and ('ObsError' in f): #### this is for current LETKF/GETKF cuz not EffectiveQC is available.
                    obserror = f['ObsError'][diag_var][:]
                    obs_vector['qc'] = qc_from_obserror(obserror, check_range=True)
    
                # Spread
                if nmember > 1:
                    hofx_ens = []
                    for iens in range(nmember):
                        hofx = f[f'hofx0_{iens+1}'][diag_var][:]
                        hofx_ens.append(hofx)
                    hofx_ens = np.stack(hofx_ens, axis=0)
                    obs_vector['spread'] = np.nanstd(hofx_ens, axis=0, ddof=0)
                    # total spread
                    obserr = f['ObsError'][diag_var][:]
                    obs_vector['totSpread'] = np.sqrt(np.nan_to_num(obs_vector['spread'], nan=0.)**2+
                        np.nan_to_num(obserr, nan=0.0)**2.0)
                    
            #### Some special cases
            if diag_obs in ["sondes", "profiler"]:
                obs_type = f['ObsType'][diag_var][:]
                if diag_obs == 'sondes':
                    obs_vector['qc'] = np.where(obs_type < 221, obs_vector['qc'], np.nan)
                else:
                    obs_vector['qc'] = np.where(obs_type == 221, obs_vector['qc'], np.nan)
                    
            if (only_qc == False):
                if diag_var == "specificHumidity":
                    obs_vector['innov'] *= 1000.
                    if nmember > 1:
                        obs_vector['spread'] *= 1000.
                        obs_vector['totSpread'] *= 1000.

                # GNSSRO: relative innovation (O-B)/O*100 [%], spreads relative to O too
                if diag_var in GNSSRO_RELATIVE_VARS:
                    obsvalue = np.where(obs_vector['obsvalue'] != 0., obs_vector['obsvalue'], np.nan)
                    obs_vector['innov'] = obs_vector['innov'] / obsvalue * 100.
                    if nmember > 1:
                        obs_vector['spread'] = obs_vector['spread'] / np.abs(obsvalue) * 100.
                        obs_vector['totSpread'] = obs_vector['totSpread'] / np.abs(obsvalue) * 100.
    
            return obs_vector
        
    except Exception as e:
        
        if filename is not None: # ignore the error message if filename = None
            print(f"Error loading {filename}: {str(e)}", file=sys.stderr)
            traceback.print_exc()
            
        return None      



def calculate_stats(data: np.ndarray, qc: np.ndarray, common_qc: np.ndarray,
                  bin_values: np.ndarray, levels: np.ndarray,
                  bin_method: int, interval: float = None) -> Dict:
    
    """Calculate statistics for each level"""
    if interval is None:
        interval = 10.0 if bin_method == 3 else (2.0 if bin_method == 2 else None)

    results = {
        'rmse': np.zeros(len(levels)+1),
        'mean': np.zeros(len(levels)+1),
        'std': np.zeros(len(levels)+1),
        'count': np.zeros(len(levels)+1),
        'rmse_common': np.zeros(len(levels)+1),
        'mean_common': np.zeros(len(levels)+1),
        'std_common': np.zeros(len(levels)+1),
        'count_common': np.zeros(len(levels)+1)
    }

    # For total
    valid_mask = (qc == 0) ##& np.isfinite(data)
    valid_data = data[valid_mask]
    results['rmse'][-1] = np.sqrt(np.nanmean(valid_data**2)) if valid_data.size else np.nan
    results['mean'][-1] = np.nanmean(valid_data) if valid_data.size else np.nan
    results['std'][-1] = np.nanstd(valid_data) if valid_data.size else np.nan
    results['count'][-1] = np.sum(~np.isnan(valid_data))

    common_mask = (common_qc == True)
    #common_mask = (qc == 0)
    common_data = data[common_mask]
    results['rmse_common'][-1] = np.sqrt(np.nanmean(common_data**2)) if common_data.size else np.nan
    results['mean_common'][-1] = np.nanmean(common_data) if common_data.size else np.nan
    results['std_common'][-1] = np.nanstd(common_data) if common_data.size else np.nan
    results['count_common'][-1] = np.sum(~np.isnan(common_data))    

    for ilev, level in enumerate(levels):
        if bin_method == 1:  # pressure
            log_levels = np.log(levels)
            if ilev == 0:
                top_bin = 200000.
                bot_bin = np.exp((log_levels[ilev] + log_levels[ilev + 1]) / 2.0)
            elif ilev == (len(levels) - 1):
                top_bin = np.exp((log_levels[ilev] + log_levels[ilev - 1]) / 2.0)
                bot_bin = 0.
            else:
                top_bin = np.exp((log_levels[ilev] + log_levels[ilev - 1]) / 2.0)
                bot_bin = np.exp((log_levels[ilev] + log_levels[ilev + 1]) / 2.0)
        else:
            bot_bin = level - interval/2.
            top_bin = level + interval/2.

        mask = (bin_values > bot_bin) & (bin_values <= top_bin) & (valid_mask)
        valid_data = data[mask]
        
        if np.any(~np.isnan(valid_data)):
            results['rmse'][ilev] = np.sqrt(np.nanmean(valid_data**2))
            results['mean'][ilev] = np.nanmean(valid_data)
            results['std'][ilev] = np.nanstd(valid_data)
            results['count'][ilev] = np.sum(~np.isnan(valid_data))
        else:
            results['rmse'][ilev] = np.nan
            results['mean'][ilev] = np.nan
            results['std'][ilev] = np.nan
            results['count'][ilev] = 0

        mask_common = (bin_values > bot_bin) & (bin_values <= top_bin) & (common_mask)
        common_data = data[mask_common]

        if np.any(~np.isnan(common_data)):
            results['rmse_common'][ilev] = np.sqrt(np.nanmean(common_data**2))
            results['mean_common'][ilev] = np.nanmean(common_data)
            results['std_common'][ilev] = np.nanstd(common_data)
            results['count_common'][ilev] = np.sum(~np.isnan(common_data))
        else:
            results['rmse_common'][ilev] = np.nan
            results['mean_common'][ilev] = np.nan
            results['std_common'][ilev] = np.nan
            results['count_common'][ilev] = 0

    return results

def process_experiment(
    config: Dict,
    diag_obs: str,
    start_time: datetime,
    end_time: datetime,
    interval: timedelta,
) -> Dict:
    """Process diagnostics for a single observation type"""

    obs_config = odc.get_obs_config(diag_obs)
    bin_method = odc.get_bin_method(diag_obs)

    if diag_obs.startswith("radar"):
        levels = odc.get_levels(bin_method, bin_width=0.5, bin_bottom=0.5, bin_top=12.0)
    else:
        levels = odc.get_levels(bin_method)

    # --------------------------------------------------
    # Domains
    # --------------------------------------------------
    domain_keys = config["plot_settings"].get("apply_domains", ["global"])
    domains = {k: config["domains"][k] for k in domain_keys if k in config["domains"]}
    if not domains:
        domains = {"global": [None, None, None, None]}

    # --------------------------------------------------
    # Time list
    # --------------------------------------------------
    times = []
    t = start_time
    while t <= end_time:
        times.append(t)
        t += interval

    num_exp = len(config["experiments"]["names"])
    num_dom = len(domains)
    num_time = len(times)
    num_var = len(obs_config.variables)
    num_lev = len(levels)

    # --------------------------------------------------
    # Allocate results
    # --------------------------------------------------
    def nan_array(*shape):
        return np.full(shape, np.nan)

    results = {
        # OMB
        "rmse_omb": nan_array(num_exp, num_dom, num_time, num_var),
        "mean_omb": nan_array(num_exp, num_dom, num_time, num_var),
        "std_omb":  nan_array(num_exp, num_dom, num_time, num_var),
        "rmse_omb_nobc": nan_array(num_exp, num_dom, num_time, num_var),
        "mean_omb_nobc": nan_array(num_exp, num_dom, num_time, num_var),
        "std_omb_nobc":  nan_array(num_exp, num_dom, num_time, num_var),
        "count_omb": nan_array(num_exp, num_dom, num_time, num_var),
        "spread_omb": nan_array(num_exp, num_dom, num_time, num_var),
        "totSpread_omb": nan_array(num_exp, num_dom, num_time, num_var),

        "rmse_omb_common": nan_array(num_exp, num_dom, num_time, num_var),
        "mean_omb_common": nan_array(num_exp, num_dom, num_time, num_var),
        "std_omb_common":  nan_array(num_exp, num_dom, num_time, num_var),
        "rmse_omb_nobc_common": nan_array(num_exp, num_dom, num_time, num_var),
        "mean_omb_nobc_common": nan_array(num_exp, num_dom, num_time, num_var),
        "std_omb_nobc_common":  nan_array(num_exp, num_dom, num_time, num_var),
        "count_omb_common": nan_array(num_exp, num_dom, num_time, num_var),
        
        "rmse_omb_bin": nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        "mean_omb_bin": nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        "std_omb_bin":  nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        "rmse_omb_nobc_bin": nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        "mean_omb_nobc_bin": nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        "std_omb_nobc_bin":  nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        "count_omb_bin": nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        "spread_omb_bin": nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        "totSpread_omb_bin": nan_array(num_exp, num_dom, num_time, num_lev, num_var),

        "rmse_omb_bin_common": nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        "mean_omb_bin_common": nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        "std_omb_bin_common":  nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        "rmse_omb_nobc_bin_common": nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        "mean_omb_nobc_bin_common": nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        "std_omb_nobc_bin_common":  nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        "count_omb_bin_common": nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        
        # OMA
        "rmse_oma": nan_array(num_exp, num_dom, num_time, num_var),
        "mean_oma": nan_array(num_exp, num_dom, num_time, num_var),
        "std_oma":  nan_array(num_exp, num_dom, num_time, num_var),
        "rmse_oma_nobc": nan_array(num_exp, num_dom, num_time, num_var),
        "mean_oma_nobc": nan_array(num_exp, num_dom, num_time, num_var),
        "std_oma_nobc":  nan_array(num_exp, num_dom, num_time, num_var),
        "count_oma": nan_array(num_exp, num_dom, num_time, num_var),
        "spread_oma": nan_array(num_exp, num_dom, num_time, num_var),
        "totSpread_oma": nan_array(num_exp, num_dom, num_time, num_var),

        "rmse_oma_common": nan_array(num_exp, num_dom, num_time, num_var),
        "mean_oma_common": nan_array(num_exp, num_dom, num_time, num_var),
        "std_oma_common":  nan_array(num_exp, num_dom, num_time, num_var),
        "rmse_oma_nobc_common": nan_array(num_exp, num_dom, num_time, num_var),
        "mean_oma_nobc_common": nan_array(num_exp, num_dom, num_time, num_var),
        "std_oma_nobc_common":  nan_array(num_exp, num_dom, num_time, num_var),
        "count_oma_common": nan_array(num_exp, num_dom, num_time, num_var),
        
        "rmse_oma_bin": nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        "mean_oma_bin": nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        "std_oma_bin":  nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        "rmse_oma_nobc_bin": nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        "mean_oma_nobc_bin": nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        "std_oma_nobc_bin":  nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        "count_oma_bin": nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        "spread_oma_bin": nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        "totSpread_oma_bin": nan_array(num_exp, num_dom, num_time, num_lev, num_var),

        "rmse_oma_bin_common": nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        "mean_oma_bin_common": nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        "std_oma_bin_common":  nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        "rmse_oma_nobc_bin_common": nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        "mean_oma_nobc_bin_common": nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        "std_oma_nobc_bin_common":  nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        "count_oma_bin_common": nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        
        # Metadata
        "times": times,
        "levels": levels,
        "variables": obs_config.variables,
        "units": obs_config.units,
        "labels": obs_config.labels,
        "domain_names": list(domains.keys()),
        "domain_coords": list(domains.values()),
    }

    # ==================================================
    # Main loop
    # ==================================================

    # `mode` (and therefore base_dir) applies to every experiment in this run --
    # mixing "da" and "hofx" experiments within one run is not supported.
    mode = config["mode"]
    base_dir = config["paths"]["exp_dir"]
    # require identical ObsValue across experiments for an obs to count as common
    check_obsvalue = config["stats_settings"].get("check_obsvalue", True)
    # per-experiment Verification/ sub-directories (hofx-verify-* modes)
    verify_settings = [odc.get_verify_settings(config, iexp)
                       for iexp in range(len(config["experiments"]["names"]))]

    for it, cycle_time in enumerate(times):

        if np.mod(it,6)==0:
            print(f'processing time t= {cycle_time}')

        for iv, diag_var in enumerate(obs_config.variables): # loop over obsvars within each obs type

            ### first step :: define the common_qc
            # Reset for every (cycle, variable) -- otherwise a missing file in the
            # first available experiment would silently reuse the previous
            # cycle's/variable's common_qc.
            omb_common_qc = None
            oma_common_qc = None
            # ObsValue of the first experiment that has data, used to check that
            # every other experiment really holds the *same* observations
            omb_ref_obsvalue = None
            oma_ref_obsvalue = None

            for iexp, (exp_name, nmember) in enumerate(
                zip(
                    config["experiments"]["names"],
                    config["experiments"]["n_member"],
                )
            ): # iexp = 0 --> reference experiment
                # HofX-application files have no outer-loop iteration groups
                outerloop_ana = 0 if odc.is_hofx_output(mode) else config["experiments"]["outerloop"][iexp]

                filedict = build_filename(
                    base_dir, exp_name, mode, nmember, cycle_time, diag_obs,
                    verify=verify_settings[iexp],
                )

                # ---------------------------
                # Load OMB / OMA (qc data only)
                # ---------------------------
                
                omb_vec = load_observation_data(
                    filedict["bak"],
                    diag_obs,
                    diag_var,
                    bin_method,
                    outerloop=0,
                    nmember=nmember, only_qc = True
                )

                if filedict["ana"]:
                    oma_vec = load_observation_data(
                        filedict["ana"],
                        diag_obs,
                        diag_var,
                        bin_method,
                        outerloop=outerloop_ana,
                        nmember=nmember, only_qc = True
                    )
                else:
                    oma_vec = None

                if omb_vec is None:
                    continue

                # select the "qualified observations" used for comparison:
                omb_qc_selected = np.isin(omb_vec['qc'], config['stats_settings']['qc_flag'])

                if omb_common_qc is None: # first experiment with data at this cycle
                    omb_common_qc = omb_qc_selected
                    omb_ref_obsvalue = omb_vec['obsvalue']
                else:
                    omb_common_qc = combine_common_qc(omb_common_qc, omb_qc_selected,
                                                      omb_ref_obsvalue, omb_vec['obsvalue'],
                                                      check_obsvalue, f"{exp_name} OMB {cycle_time} {diag_var}")

                # OMA doesn't exist in hofx / hofx-verify-bg modes
                if oma_vec is not None:
                    oma_qc_selected = np.isin(oma_vec['qc'], config['stats_settings']['qc_flag'])

                    if oma_common_qc is None:
                        oma_common_qc = oma_qc_selected
                        oma_ref_obsvalue = oma_vec['obsvalue']
                    else:
                        oma_common_qc = combine_common_qc(oma_common_qc, oma_qc_selected,
                                                          oma_ref_obsvalue, oma_vec['obsvalue'],
                                                          check_obsvalue, f"{exp_name} OMA {cycle_time} {diag_var}")

            ### Second step :: original code for calculating the stats:
            for iexp, (exp_name, nmember) in enumerate(
                zip(
                    config["experiments"]["names"],
                    config["experiments"]["n_member"],
                )
            ): # iexp = 0 --> reference experiment

                # HofX-application files have no outer-loop iteration groups
                outerloop_ana = 0 if odc.is_hofx_output(mode) else config["experiments"]["outerloop"][iexp]

                filedict = build_filename(
                    base_dir, exp_name, mode, nmember, cycle_time, diag_obs,
                    verify=verify_settings[iexp],
                )
                
                # ---------------------------
                # Load OMB / OMA (all stats)
                # ---------------------------                
                omb_vec = load_observation_data(
                    filedict["bak"],
                    diag_obs,
                    diag_var,
                    bin_method,
                    outerloop=0,
                    nmember=nmember,
                )

                if filedict["ana"]:
                    oma_vec = load_observation_data(
                        filedict["ana"],
                        diag_obs,
                        diag_var,
                        bin_method,
                        outerloop=outerloop_ana,
                        nmember=nmember,
                    )
                else:
                    oma_vec = None

                if omb_vec is None:
                    continue

                # add the common_qc information (an experiment whose nlocs differs from
                # the reference has no common obs -- see combine_common_qc):
                nlocs_omb = len(omb_vec['lat'])
                omb_vec['common_qc'] = (omb_common_qc if len(omb_common_qc) == nlocs_omb
                                        else np.zeros(nlocs_omb, dtype=bool))
                if oma_vec is not None:
                    nlocs_oma = len(oma_vec['lat'])
                    oma_vec['common_qc'] = (oma_common_qc if (oma_common_qc is not None and len(oma_common_qc) == nlocs_oma)
                                            else np.zeros(nlocs_oma, dtype=bool))

                # ---------------------------
                # Bin coordinate
                # ---------------------------
                bin_values = (
                    omb_vec["pres"] if bin_method == 1 else
                    omb_vec["hgt"]  if bin_method == 2 else
                    omb_vec["lat"]
                )

                # ---------------------------
                # Domain loop
                # ---------------------------
                for idom, (dom_name, dom_coords) in enumerate(domains.items()):

                    def apply_domain_mask(qc, lat, lon):
                        if qc is None:
                            return None
                        if dom_name == "global":
                            return qc
                        lat_min, lat_max, lon_min, lon_max = dom_coords
                        mask = (
                            (lat >= lat_min) & (lat <= lat_max) &
                            (lon >= lon_min) & (lon <= lon_max)
                        )
                        return np.where(mask, qc, np.nan)

                    omb_qc = apply_domain_mask(
                        omb_vec["qc"], omb_vec["lat"], omb_vec["lon"]
                    )
                    oma_qc = (
                        apply_domain_mask(
                            oma_vec["qc"], oma_vec["lat"], oma_vec["lon"]
                        )
                        if oma_vec is not None
                        else None
                    )

                    # NOTE: use *_dom names here (not omb_common_qc/oma_common_qc) so we
                    # don't clobber the cross-experiment common_qc computed in the first
                    # pass above -- that variable is re-read on the next iexp iteration
                    # (see `omb_vec['common_qc'] = omb_common_qc` before this domain loop).
                    omb_common_qc_dom = apply_domain_mask(
                        omb_vec["common_qc"], omb_vec["lat"], omb_vec["lon"]
                    )
                    oma_common_qc_dom = (
                        apply_domain_mask(
                            oma_vec["common_qc"], oma_vec["lat"], oma_vec["lon"]
                        )
                        if oma_vec is not None
                        else None
                    )

                    
                    # =======================
                    # OMB stats
                    # =======================
                    omb_stats = calculate_stats(
                        omb_vec["innov"],
                        omb_qc,
                        omb_common_qc_dom,
                        bin_values,
                        levels,
                        bin_method,
                    )

                    if omb_vec["innov_nobc"] is not None:
                        omb_nobc_stats = calculate_stats(
                                            omb_vec["innov_nobc"],
                                            omb_qc,
                                            omb_common_qc_dom,
                                            bin_values,
                                            levels,
                                            bin_method,
                                        )

                    # total statistics (without binning)
                    if omb_stats["count"][num_lev] > 0:
                        results["rmse_omb"][iexp, idom, it, iv] = omb_stats["rmse"][num_lev]
                        results["mean_omb"][iexp, idom, it, iv] = omb_stats["mean"][num_lev]
                        results["std_omb"][iexp, idom, it, iv]  = omb_stats["std"][num_lev]
                        results["count_omb"][iexp, idom, it, iv] = omb_stats["count"][num_lev]
                        if omb_vec["innov_nobc"] is not None:
                            results["rmse_omb_nobc"][iexp, idom, it, iv] = omb_nobc_stats["rmse"][num_lev]
                            results["mean_omb_nobc"][iexp, idom, it, iv] = omb_nobc_stats["mean"][num_lev]
                            results["std_omb_nobc"][iexp, idom, it, iv]  = omb_nobc_stats["std"][num_lev]                            

                    if omb_stats["count_common"][num_lev] > 0:
                        results["rmse_omb_common"][iexp, idom, it, iv] = omb_stats["rmse_common"][num_lev]
                        results["mean_omb_common"][iexp, idom, it, iv] = omb_stats["mean_common"][num_lev]
                        results["std_omb_common"][iexp, idom, it, iv]  = omb_stats["std_common"][num_lev]
                        results["count_omb_common"][iexp, idom, it, iv] = omb_stats["count_common"][num_lev]
                        if omb_vec["innov_nobc"] is not None:
                            results["rmse_omb_nobc_common"][iexp, idom, it, iv] = omb_nobc_stats["rmse_common"][num_lev]
                            results["mean_omb_nobc_common"][iexp, idom, it, iv] = omb_nobc_stats["mean_common"][num_lev]
                            results["std_omb_nobc_common"][iexp, idom, it, iv]  = omb_nobc_stats["std_common"][num_lev]                            

                    valid = omb_stats["count"][:num_lev] > 0
                    results["rmse_omb_bin"][iexp, idom, it, valid, iv] = omb_stats["rmse"][:num_lev][valid]
                    results["mean_omb_bin"][iexp, idom, it, valid, iv] = omb_stats["mean"][:num_lev][valid]
                    results["std_omb_bin"][iexp, idom, it, valid, iv]  = omb_stats["std"][:num_lev][valid]
                    results["count_omb_bin"][iexp, idom, it, valid, iv] = omb_stats["count"][:num_lev][valid]
                    if omb_vec["innov_nobc"] is not None:
                        results["rmse_omb_nobc_bin"][iexp, idom, it, valid, iv] = omb_nobc_stats["rmse"][:num_lev][valid]
                        results["mean_omb_nobc_bin"][iexp, idom, it, valid, iv] = omb_nobc_stats["mean"][:num_lev][valid]
                        results["std_omb_nobc_bin"][iexp, idom, it, valid, iv]  = omb_nobc_stats["std"][:num_lev][valid] 

                    valid_common = omb_stats["count_common"][:num_lev] > 0
                    results["rmse_omb_bin_common"][iexp, idom, it, valid_common, iv] = omb_stats["rmse_common"][:num_lev][valid_common]
                    results["mean_omb_bin_common"][iexp, idom, it, valid_common, iv] = omb_stats["mean_common"][:num_lev][valid_common]
                    results["std_omb_bin_common"][iexp, idom, it, valid_common, iv]  = omb_stats["std_common"][:num_lev][valid_common]
                    results["count_omb_bin_common"][iexp, idom, it, valid_common, iv] = omb_stats["count_common"][:num_lev][valid_common]
                    if omb_vec["innov_nobc"] is not None:
                        results["rmse_omb_nobc_bin_common"][iexp, idom, it, valid_common, iv] = omb_nobc_stats["rmse_common"][:num_lev][valid_common]
                        results["mean_omb_nobc_bin_common"][iexp, idom, it, valid_common, iv] = omb_nobc_stats["mean_common"][:num_lev][valid_common]
                        results["std_omb_nobc_bin_common"][iexp, idom, it, valid_common, iv]  = omb_nobc_stats["std_common"][:num_lev][valid_common] 

                        
                    if omb_vec["spread"] is not None:
                        omb_stats = calculate_stats(
                            omb_vec["spread"],
                            omb_qc,
                            omb_common_qc_dom,
                            bin_values,
                            levels,
                            bin_method,
                        )
                        if omb_stats["count"][num_lev] > 0:
                            results["spread_omb"][iexp, idom, it, iv] = omb_stats["mean"][num_lev]
                        valid = omb_stats["count"][:num_lev] > 0
                        results["spread_omb_bin"][iexp, idom, it, valid, iv] = omb_stats["mean"][:num_lev][valid]

                    if omb_vec["totSpread"] is not None:
                        omb_stats = calculate_stats(
                            omb_vec["totSpread"],
                            omb_qc,
                            omb_common_qc_dom,
                            bin_values,
                            levels,
                            bin_method,
                        )
                        if omb_stats["count"][num_lev] > 0:
                            results["totSpread_omb"][iexp, idom, it, iv] = omb_stats["mean"][num_lev]
                        valid = omb_stats["count"][:num_lev] > 0
                        results["totSpread_omb_bin"][iexp, idom, it, valid, iv] = omb_stats["mean"][:num_lev][valid]

                    # =======================
                    # OMA stats
                    # =======================
                    
                    if oma_vec is not None and oma_vec["innov"] is not None:
                        oma_stats = calculate_stats(
                            oma_vec["innov"],
                            oma_qc,
                            oma_common_qc_dom,
                            bin_values,
                            levels,
                            bin_method,
                        )

                        if oma_vec["innov_nobc"] is not None:
                            oma_nobc_stats = calculate_stats(
                                                oma_vec["innov_nobc"],
                                                oma_qc,
                                                oma_common_qc_dom,
                                                bin_values,
                                                levels,
                                                bin_method,
                                            )
                        
                        # total statistics (without binning)
                        if oma_stats["count"][num_lev] > 0:
                            results["rmse_oma"][iexp, idom, it, iv] = oma_stats["rmse"][num_lev]
                            results["mean_oma"][iexp, idom, it, iv] = oma_stats["mean"][num_lev]
                            results["std_oma"][iexp, idom, it, iv]  = oma_stats["std"][num_lev]
                            results["count_oma"][iexp, idom, it, iv] = oma_stats["count"][num_lev]
                            if oma_vec["innov_nobc"] is not None:
                                results["rmse_oma_nobc"][iexp, idom, it, iv] = oma_nobc_stats["rmse"][num_lev]
                                results["mean_oma_nobc"][iexp, idom, it, iv] = oma_nobc_stats["mean"][num_lev]
                                results["std_oma_nobc"][iexp, idom, it, iv]  = oma_nobc_stats["std"][num_lev] 

                        if oma_stats["count_common"][num_lev] > 0:
                            results["rmse_oma_common"][iexp, idom, it, iv] = oma_stats["rmse_common"][num_lev]
                            results["mean_oma_common"][iexp, idom, it, iv] = oma_stats["mean_common"][num_lev]
                            results["std_oma_common"][iexp, idom, it, iv]  = oma_stats["std_common"][num_lev]
                            results["count_oma_common"][iexp, idom, it, iv] = oma_stats["count_common"][num_lev]
                            if oma_vec["innov_nobc"] is not None:
                                results["rmse_oma_nobc_common"][iexp, idom, it, iv] = oma_nobc_stats["rmse_common"][num_lev]
                                results["mean_oma_nobc_common"][iexp, idom, it, iv] = oma_nobc_stats["mean_common"][num_lev]
                                results["std_oma_nobc_common"][iexp, idom, it, iv]  = oma_nobc_stats["std_common"][num_lev]     

                        valid = oma_stats["count"][:num_lev] > 0
                        results["rmse_oma_bin"][iexp, idom, it, valid, iv] = oma_stats["rmse"][:num_lev][valid]
                        results["mean_oma_bin"][iexp, idom, it, valid, iv] = oma_stats["mean"][:num_lev][valid]
                        results["std_oma_bin"][iexp, idom, it, valid, iv]  = oma_stats["std"][:num_lev][valid]
                        results["count_oma_bin"][iexp, idom, it, valid, iv] = oma_stats["count"][:num_lev][valid]
                        if oma_vec["innov_nobc"] is not None:
                            results["rmse_oma_nobc_bin"][iexp, idom, it, valid, iv] = oma_nobc_stats["rmse"][:num_lev][valid]
                            results["mean_oma_nobc_bin"][iexp, idom, it, valid, iv] = oma_nobc_stats["mean"][:num_lev][valid]
                            results["std_oma_nobc_bin"][iexp, idom, it, valid, iv]  = oma_nobc_stats["std"][:num_lev][valid] 
                            
                        valid_common = oma_stats["count_common"][:num_lev] > 0
                        results["rmse_oma_bin_common"][iexp, idom, it, valid_common, iv] = oma_stats["rmse_common"][:num_lev][valid_common]
                        results["mean_oma_bin_common"][iexp, idom, it, valid_common, iv] = oma_stats["mean_common"][:num_lev][valid_common]
                        results["std_oma_bin_common"][iexp, idom, it, valid_common, iv]  = oma_stats["std_common"][:num_lev][valid_common]
                        results["count_oma_bin_common"][iexp, idom, it, valid_common, iv] = oma_stats["count_common"][:num_lev][valid_common]
                        if oma_vec["innov_nobc"] is not None:
                            results["rmse_oma_nobc_bin_common"][iexp, idom, it, valid_common, iv] = oma_nobc_stats["rmse_common"][:num_lev][valid_common]
                            results["mean_oma_nobc_bin_common"][iexp, idom, it, valid_common, iv] = oma_nobc_stats["mean_common"][:num_lev][valid_common]
                            results["std_oma_nobc_bin_common"][iexp, idom, it, valid_common, iv]  = oma_nobc_stats["std_common"][:num_lev][valid_common]
                            
                        if oma_vec["spread"] is not None:
                            oma_stats = calculate_stats(
                                oma_vec["spread"],
                                oma_qc,
                                oma_common_qc_dom,
                                bin_values,
                                levels,
                                bin_method,
                            )
                            if oma_stats["count"][num_lev] > 0:
                                results["spread_oma"][iexp, idom, it, iv] = oma_stats["mean"][num_lev]
                            valid = oma_stats["count"][:num_lev] > 0
                            results["spread_oma_bin"][iexp, idom, it, valid, iv] = oma_stats["mean"][:num_lev][valid]

                        if oma_vec["totSpread"] is not None:
                            oma_stats = calculate_stats(
                                oma_vec["totSpread"],
                                oma_qc,
                                oma_common_qc_dom,
                                bin_values,
                                levels,
                                bin_method,
                            )
                            if oma_stats["count"][num_lev] > 0:
                                results["totSpread_oma"][iexp, idom, it, iv] = oma_stats["mean"][num_lev]
                            valid = oma_stats["count"][:num_lev] > 0
                            results["totSpread_oma_bin"][iexp, idom, it, valid, iv] = oma_stats["mean"][:num_lev][valid]

    return results




def remove_outliers(data, threshold=3):
    """Drop points more than `threshold` standard deviations from the mean
    (simple z-score filter used before computing CIs)."""
    data = np.asarray(data)
    clean = data[np.isfinite(data)]
    if clean.size == 0: return clean
    mean, std = np.mean(clean), np.std(clean)
    if std == 0: return clean
    z = np.abs((clean - mean) / std)
    return clean[z < threshold]


def ttest_CI(data, alpha=0.95):
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return (np.nan, np.nan, np.nan)
    data = remove_outliers(data)  
    
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    se = std / np.sqrt(n)
    tcrit = stats.t.ppf((1 + alpha) / 2, df=n-1)

    lower = mean - tcrit * se
    upper = mean + tcrit * se
    return mean, upper, lower


def bootstrap_CI(data, alpha=0.95, n_samples=8000):
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return (np.nan, np.nan, np.nan)
    data = remove_outliers(data)
    rng = np.random.default_rng()
    n = data.size
    boot = np.empty(n_samples)
    for i in range(n_samples):
        idx = rng.integers(0, n, size=n)
        boot[i] = np.mean(data[idx])
    lower = np.percentile(boot, (1-alpha)/2*100)
    upper = np.percentile(boot, (alpha+(1-alpha)/2)*100)
    return np.mean(data), upper, lower


def compute_ci(data_test: np.ndarray, data_ref: np.ndarray, ci_method: str):
    """Normalized percent difference of data_test vs. data_ref (100 = no
    change), with a confidence interval from either method. Returns
    (mean_diff, ci_upper, ci_lower).
    """
    normalized_diff = 100 + (data_test - data_ref) / data_ref * 100
    if ci_method == 'paired_t_test':
        return ttest_CI(normalized_diff)
    elif ci_method == 'bootstrap':
        return bootstrap_CI(normalized_diff)
    raise ValueError(f"unknown ci_method: {ci_method!r} (expected 'paired_t_test' or 'bootstrap')")

def normalized_exp_diff_stats(config, results, ci_method='paired_t_test'):
    #ci_method = 'paired_t_test'
    #ci_method = 'bootstrap'

    # Derive dimensions from `results` instead of relying on the module-level
    # globals that process_experiment() sets as a side effect -- those are only
    # valid if this is called right after process_experiment() for the same
    # diag_obs, which is fragile (e.g. breaks if the sensor loop is reordered).
    num_exp = results['rmse_omb'].shape[0]
    num_dom = len(results['domain_names'])
    num_var = len(results['variables'])
    num_lev = len(results['levels'])

    def nan_array(*shape):
        return np.full(shape, np.nan)
    
    # Allocate one nan_array per (omb/oma) x (rmse/std) x (bias-corrected/nobc) x
    # (mean-diff/ci_upper/ci_lower) x (unbinned/binned) combination, plus a plain
    # `count` per (omb/oma) x (unbinned/binned). This produces the same 52 keys
    # that used to be written out by hand (e.g. 'md_rmse_omb', 'ci_upper_rmse_omb_nobc_bin').
    norm_results = {}
    for omboma in ('omb', 'oma'):
        norm_results[f'count_{omboma}']     = nan_array(num_exp-1, num_dom, num_var)
        norm_results[f'count_{omboma}_bin'] = nan_array(num_exp-1, num_dom, num_lev, num_var)
        for metric in ('rmse', 'std'):
            for nobc in ('', '_nobc'):
                base = f'{metric}_{omboma}{nobc}'
                for stat in ('md', 'ci_upper', 'ci_lower'):
                    norm_results[f'{stat}_{base}']     = nan_array(num_exp-1, num_dom, num_var)
                    norm_results[f'{stat}_{base}_bin'] = nan_array(num_exp-1, num_dom, num_lev, num_var)

    for iexp, exp_name in enumerate(config["experiments"]["names"]): # iexp = 0 --> reference experiment 
        if iexp > 0: # iexp=0: reference experiment
    
            for idom in range(num_dom):
                for ivar in range(num_var):
                    # un-binned variables:
                    norm_results['count_omb'][iexp-1, idom, ivar] = np.nanmean(results['count_omb_common'][0, idom, :, ivar])
                    norm_results['count_oma'][iexp-1, idom, ivar] = np.nanmean(results['count_oma_common'][0, idom, :, ivar])
                    
                    # omb - rmse
                    data_test = results['rmse_omb_common'][iexp, idom, :, ivar]
                    data_ref  = results['rmse_omb_common'][0, idom, :, ivar]
                    
                    meandiff, ci_upper, ci_lower = compute_ci(data_test, data_ref, ci_method)
     
                    norm_results['md_rmse_omb'][iexp-1, idom, ivar] = meandiff
                    norm_results['ci_upper_rmse_omb'][iexp-1, idom, ivar] = ci_upper
                    norm_results['ci_lower_rmse_omb'][iexp-1, idom, ivar] = ci_lower

                    # omb - rmse (nobc)
                    if not np.all(np.isnan(results['rmse_omb_nobc_common'])):               
                        data_test = results['rmse_omb_nobc_common'][iexp, idom, :, ivar]
                        data_ref  = results['rmse_omb_nobc_common'][0, idom, :, ivar]
                        
                        meandiff, ci_upper, ci_lower = compute_ci(data_test, data_ref, ci_method)
         
                        norm_results['md_rmse_omb_nobc'][iexp-1, idom, ivar] = meandiff
                        norm_results['ci_upper_rmse_omb_nobc'][iexp-1, idom, ivar] = ci_upper
                        norm_results['ci_lower_rmse_omb_nobc'][iexp-1, idom, ivar] = ci_lower

                    # omb - std
                    data_test = results['std_omb_common'][iexp, idom, :, ivar]
                    data_ref  = results['std_omb_common'][0, idom, :, ivar]
                    
                    meandiff, ci_upper, ci_lower = compute_ci(data_test, data_ref, ci_method)
     
                    norm_results['md_std_omb'][iexp-1, idom, ivar] = meandiff
                    norm_results['ci_upper_std_omb'][iexp-1, idom, ivar] = ci_upper
                    norm_results['ci_lower_std_omb'][iexp-1, idom, ivar] = ci_lower

                    # omb - std (nobc)
                    if not np.all(np.isnan(results['std_omb_nobc_common'])):               
                        data_test = results['std_omb_nobc_common'][iexp, idom, :, ivar]
                        data_ref  = results['std_omb_nobc_common'][0, idom, :, ivar]
                        
                        meandiff, ci_upper, ci_lower = compute_ci(data_test, data_ref, ci_method)
         
                        norm_results['md_std_omb_nobc'][iexp-1, idom, ivar] = meandiff
                        norm_results['ci_upper_std_omb_nobc'][iexp-1, idom, ivar] = ci_upper
                        norm_results['ci_lower_std_omb_nobc'][iexp-1, idom, ivar] = ci_lower
                    
                    # oma - rmse
                    data_test = results['rmse_oma_common'][iexp, idom, :, ivar]
                    data_ref  = results['rmse_oma_common'][0, idom, :, ivar]
                    
                    meandiff, ci_upper, ci_lower = compute_ci(data_test, data_ref, ci_method)
     
                    norm_results['md_rmse_oma'][iexp-1, idom, ivar] = meandiff
                    norm_results['ci_upper_rmse_oma'][iexp-1, idom, ivar] = ci_upper
                    norm_results['ci_lower_rmse_oma'][iexp-1, idom, ivar] = ci_lower

                    # oma - rmse (nobc)
                    if not np.all(np.isnan(results['rmse_oma_nobc_common'])):               
                        data_test = results['rmse_oma_nobc_common'][iexp, idom, :, ivar]
                        data_ref  = results['rmse_oma_nobc_common'][0, idom, :, ivar]
                        
                        meandiff, ci_upper, ci_lower = compute_ci(data_test, data_ref, ci_method)
         
                        norm_results['md_rmse_oma_nobc'][iexp-1, idom, ivar] = meandiff
                        norm_results['ci_upper_rmse_oma_nobc'][iexp-1, idom, ivar] = ci_upper
                        norm_results['ci_lower_rmse_oma_nobc'][iexp-1, idom, ivar] = ci_lower
                    
                    # oma - std
                    data_test = results['std_oma_common'][iexp, idom, :, ivar]
                    data_ref  = results['std_oma_common'][0, idom, :, ivar]
                    
                    meandiff, ci_upper, ci_lower = compute_ci(data_test, data_ref, ci_method)
     
                    norm_results['md_std_oma'][iexp-1, idom, ivar] = meandiff
                    norm_results['ci_upper_std_oma'][iexp-1, idom, ivar] = ci_upper
                    norm_results['ci_lower_std_oma'][iexp-1, idom, ivar] = ci_lower

                    # oma - std (nobc)
                    if not np.all(np.isnan(results['std_oma_nobc_common'])):               
                        data_test = results['std_oma_nobc_common'][iexp, idom, :, ivar]
                        data_ref  = results['std_oma_nobc_common'][0, idom, :, ivar]
                        
                        meandiff, ci_upper, ci_lower = compute_ci(data_test, data_ref, ci_method)
         
                        norm_results['md_std_oma_nobc'][iexp-1, idom, ivar] = meandiff
                        norm_results['ci_upper_std_oma_nobc'][iexp-1, idom, ivar] = ci_upper
                        norm_results['ci_lower_std_oma_nobc'][iexp-1, idom, ivar] = ci_lower
                    
                    # for binned variables:
                    for ilev in range(num_lev):
                        norm_results['count_omb_bin'][iexp-1, idom, ilev, ivar] = np.nanmean(results['count_omb_bin_common'][0, idom, :, ilev, ivar])
                        norm_results['count_oma_bin'][iexp-1, idom, ilev, ivar] = np.nanmean(results['count_oma_bin_common'][0, idom, :, ilev, ivar])
                        
                        # omb - rmse
                        data_test = results['rmse_omb_bin_common'][iexp, idom, :, ilev, ivar]
                        data_ref  = results['rmse_omb_bin_common'][0, idom, :, ilev, ivar]
                        
                        meandiff, ci_upper, ci_lower = compute_ci(data_test, data_ref, ci_method)
         
                        norm_results['md_rmse_omb_bin'][iexp-1, idom, ilev, ivar] = meandiff
                        norm_results['ci_upper_rmse_omb_bin'][iexp-1, idom, ilev, ivar] = ci_upper
                        norm_results['ci_lower_rmse_omb_bin'][iexp-1, idom, ilev, ivar] = ci_lower

                        # omb - rmse (nobc)
                        if not np.all(np.isnan(results['rmse_omb_nobc_bin_common'])):               
                            data_test = results['rmse_omb_nobc_bin_common'][iexp, idom, :, ilev, ivar]
                            data_ref  = results['rmse_omb_nobc_bin_common'][0, idom, :, ilev, ivar]
                            
                            meandiff, ci_upper, ci_lower = compute_ci(data_test, data_ref, ci_method)
             
                            norm_results['md_rmse_omb_nobc_bin'][iexp-1, idom, ilev, ivar] = meandiff
                            norm_results['ci_upper_rmse_omb_nobc_bin'][iexp-1, idom, ilev, ivar] = ci_upper
                            norm_results['ci_lower_rmse_omb_nobc_bin'][iexp-1, idom, ilev, ivar] = ci_lower

                        
                        # omb - std
                        data_test = results['std_omb_bin_common'][iexp, idom, :, ilev, ivar]
                        data_ref  = results['std_omb_bin_common'][0, idom, :, ilev, ivar]
                        
                        meandiff, ci_upper, ci_lower = compute_ci(data_test, data_ref, ci_method)
         
                        norm_results['md_std_omb_bin'][iexp-1, idom, ilev, ivar] = meandiff
                        norm_results['ci_upper_std_omb_bin'][iexp-1, idom, ilev, ivar] = ci_upper
                        norm_results['ci_lower_std_omb_bin'][iexp-1, idom, ilev, ivar] = ci_lower

                        # omb - std (nobc)
                        if not np.all(np.isnan(results['std_omb_nobc_bin_common'])):               
                            data_test = results['std_omb_nobc_bin_common'][iexp, idom, :, ilev, ivar]
                            data_ref  = results['std_omb_nobc_bin_common'][0, idom, :, ilev, ivar]
                            
                            meandiff, ci_upper, ci_lower = compute_ci(data_test, data_ref, ci_method)
             
                            norm_results['md_std_omb_nobc_bin'][iexp-1, idom, ilev, ivar] = meandiff
                            norm_results['ci_upper_std_omb_nobc_bin'][iexp-1, idom, ilev, ivar] = ci_upper
                            norm_results['ci_lower_std_omb_nobc_bin'][iexp-1, idom, ilev, ivar] = ci_lower
                        
                        # oma - rmse
                        data_test = results['rmse_oma_bin_common'][iexp, idom, :, ilev, ivar]
                        data_ref  = results['rmse_oma_bin_common'][0, idom, :, ilev, ivar]
                        
                        meandiff, ci_upper, ci_lower = compute_ci(data_test, data_ref, ci_method)
         
                        norm_results['md_rmse_oma_bin'][iexp-1, idom, ilev, ivar] = meandiff
                        norm_results['ci_upper_rmse_oma_bin'][iexp-1, idom, ilev, ivar] = ci_upper
                        norm_results['ci_lower_rmse_oma_bin'][iexp-1, idom, ilev, ivar] = ci_lower
                        
                        # oma - rmse (nobc)
                        if not np.all(np.isnan(results['rmse_oma_nobc_bin_common'])):               
                            data_test = results['rmse_oma_nobc_bin_common'][iexp, idom, :, ilev, ivar]
                            data_ref  = results['rmse_oma_nobc_bin_common'][0, idom, :, ilev, ivar]
                            
                            meandiff, ci_upper, ci_lower = compute_ci(data_test, data_ref, ci_method)
             
                            norm_results['md_rmse_oma_nobc_bin'][iexp-1, idom, ilev, ivar] = meandiff
                            norm_results['ci_upper_rmse_oma_nobc_bin'][iexp-1, idom, ilev, ivar] = ci_upper
                            norm_results['ci_lower_rmse_oma_nobc_bin'][iexp-1, idom, ilev, ivar] = ci_lower
                            
                        # oma - std
                        data_test = results['std_oma_bin_common'][iexp, idom, :, ilev, ivar]
                        data_ref  = results['std_oma_bin_common'][0, idom, :, ilev, ivar]
                        
                        meandiff, ci_upper, ci_lower = compute_ci(data_test, data_ref, ci_method)
         
                        norm_results['md_std_oma_bin'][iexp-1, idom, ilev, ivar] = meandiff
                        norm_results['ci_upper_std_oma_bin'][iexp-1, idom, ilev, ivar] = ci_upper
                        norm_results['ci_lower_std_oma_bin'][iexp-1, idom, ilev, ivar] = ci_lower

                        # oma - std (nobc)
                        if not np.all(np.isnan(results['std_oma_nobc_bin_common'])):               
                            data_test = results['std_oma_nobc_bin_common'][iexp, idom, :, ilev, ivar]
                            data_ref  = results['std_oma_nobc_bin_common'][0, idom, :, ilev, ivar]
                            
                            meandiff, ci_upper, ci_lower = compute_ci(data_test, data_ref, ci_method)
             
                            norm_results['md_std_oma_nobc_bin'][iexp-1, idom, ilev, ivar] = meandiff
                            norm_results['ci_upper_std_oma_nobc_bin'][iexp-1, idom, ilev, ivar] = ci_upper
                            norm_results['ci_lower_std_oma_nobc_bin'][iexp-1, idom, ilev, ivar] = ci_lower

    # carry level values through so downstream csv-writers don't need module globals
    norm_results['levels'] = results['levels']

    return norm_results

def compile_norm_results_and_save_into_csv(config, diag_obs, bin_method, norm_results):

    csv_output = config['paths']['output_dir']+'/csv/'+diag_obs
    os.makedirs(csv_output, exist_ok=True)

    exp_labels = config['experiments']['labels']
    tags = odc.get_obs_config(diag_obs).tags

    # derive dimensions from norm_results instead of relying on module globals
    num_exp = norm_results['md_rmse_omb'].shape[0] + 1
    num_var = norm_results['md_rmse_omb'].shape[-1]
    levels = norm_results['levels']

    ## unbinned variables:
    for idom, dom in enumerate(config['plot_settings']['apply_domains']):
        dom_id = config['plot_settings']['apply_domains'][idom]

        norm_dic={};
        norm_dic['exp_ref'] = []; norm_dic['exp_tag'] = [];
        norm_dic['domain'] = [];
        norm_dic['var_name'] = [];  
        norm_dic['count_omb'] = []; 
        norm_dic['md_rmse_omb'] = []; norm_dic['ci_upper_rmse_omb'] = []; norm_dic['ci_lower_rmse_omb'] = [];
        norm_dic['md_std_omb'] = []; norm_dic['ci_upper_std_omb'] = []; norm_dic['ci_lower_std_omb'] = [];
        norm_dic['md_rmse_omb_nobc'] = []; norm_dic['ci_upper_rmse_omb_nobc'] = []; norm_dic['ci_lower_rmse_omb_nobc'] = [];
        norm_dic['md_std_omb_nobc'] = []; norm_dic['ci_upper_std_omb_nobc'] = []; norm_dic['ci_lower_std_omb_nobc'] = [];
        
        norm_dic['count_oma'] = []; 
        norm_dic['md_rmse_oma'] = []; norm_dic['ci_upper_rmse_oma'] = []; norm_dic['ci_lower_rmse_oma'] = [];
        norm_dic['md_std_oma'] = []; norm_dic['ci_upper_std_oma'] = []; norm_dic['ci_lower_std_oma'] = [];
        norm_dic['md_rmse_oma_nobc'] = []; norm_dic['ci_upper_rmse_oma_nobc'] = []; norm_dic['ci_lower_rmse_oma_nobc'] = [];
        norm_dic['md_std_oma_nobc'] = []; norm_dic['ci_upper_std_oma_nobc'] = []; norm_dic['ci_lower_std_oma_nobc'] = [];    
        
        for ivar in range(num_var):
            for iexp in range(num_exp-1):
                norm_dic['exp_ref'].append(exp_labels[0])
                norm_dic['exp_tag'].append(exp_labels[iexp+1])
                norm_dic['domain'].append(dom)
                norm_dic['var_name'].append(tags[ivar])
                norm_dic['count_omb'].append(norm_results['count_omb'][iexp, idom, ivar])
                norm_dic['md_rmse_omb'].append(norm_results['md_rmse_omb'][iexp, idom, ivar])
                norm_dic['ci_upper_rmse_omb'].append(norm_results['ci_upper_rmse_omb'][iexp, idom, ivar])
                norm_dic['ci_lower_rmse_omb'].append(norm_results['ci_lower_rmse_omb'][iexp, idom, ivar])
                norm_dic['md_std_omb'].append(norm_results['md_std_omb'][iexp, idom, ivar])
                norm_dic['ci_upper_std_omb'].append(norm_results['ci_upper_std_omb'][iexp, idom, ivar])
                norm_dic['ci_lower_std_omb'].append(norm_results['ci_lower_std_omb'][iexp, idom, ivar])
                norm_dic['md_rmse_omb_nobc'].append(norm_results['md_rmse_omb_nobc'][iexp, idom, ivar])
                norm_dic['ci_upper_rmse_omb_nobc'].append(norm_results['ci_upper_rmse_omb_nobc'][iexp, idom, ivar])
                norm_dic['ci_lower_rmse_omb_nobc'].append(norm_results['ci_lower_rmse_omb_nobc'][iexp, idom, ivar])
                norm_dic['md_std_omb_nobc'].append(norm_results['md_std_omb_nobc'][iexp, idom, ivar])
                norm_dic['ci_upper_std_omb_nobc'].append(norm_results['ci_upper_std_omb_nobc'][iexp, idom, ivar])
                norm_dic['ci_lower_std_omb_nobc'].append(norm_results['ci_lower_std_omb_nobc'][iexp, idom, ivar])
                
                norm_dic['count_oma'].append(norm_results['count_oma'][iexp, idom, ivar])
                norm_dic['md_rmse_oma'].append(norm_results['md_rmse_oma'][iexp, idom, ivar])
                norm_dic['ci_upper_rmse_oma'].append(norm_results['ci_upper_rmse_oma'][iexp, idom, ivar])
                norm_dic['ci_lower_rmse_oma'].append(norm_results['ci_lower_rmse_oma'][iexp, idom, ivar])
                norm_dic['md_std_oma'].append(norm_results['md_std_oma'][iexp, idom, ivar])
                norm_dic['ci_upper_std_oma'].append(norm_results['ci_upper_std_oma'][iexp, idom, ivar])
                norm_dic['ci_lower_std_oma'].append(norm_results['ci_lower_std_oma'][iexp, idom, ivar])
                norm_dic['md_rmse_oma_nobc'].append(norm_results['md_rmse_oma_nobc'][iexp, idom, ivar])
                norm_dic['ci_upper_rmse_oma_nobc'].append(norm_results['ci_upper_rmse_oma_nobc'][iexp, idom, ivar])
                norm_dic['ci_lower_rmse_oma_nobc'].append(norm_results['ci_lower_rmse_oma_nobc'][iexp, idom, ivar])
                norm_dic['md_std_oma_nobc'].append(norm_results['md_std_oma_nobc'][iexp, idom, ivar])
                norm_dic['ci_upper_std_oma_nobc'].append(norm_results['ci_upper_std_oma_nobc'][iexp, idom, ivar])
                norm_dic['ci_lower_std_oma_nobc'].append(norm_results['ci_lower_std_oma_nobc'][iexp, idom, ivar])

        # save into .csv file (moved inside the domain loop -- one file per domain,
        # previously this only ran once after the loop and silently dropped every
        # domain except the last)
        filename = f'{csv_output}/{diag_obs}_{dom}_unbinned.csv'

        df = pd.DataFrame(norm_dic)
        df.to_csv(filename, index=False)

    ## binned variables:
    if not diag_obs.startswith("sfc"):
        bin_label = {1: "pressure", 2: "height", 3: "latitude"}.get(bin_method, "level")
    
        for idom, dom in enumerate(config['plot_settings']['apply_domains']):
            dom_id = config['plot_settings']['apply_domains'][idom]
        
            norm_dic = {};
            norm_dic['exp_ref'] = []; norm_dic['exp_tag'] = [];
            norm_dic['domain'] = [];  norm_dic[f'{bin_label}'] = [];
            norm_dic['var_name'] = [];  
            norm_dic['count_omb'] = [];
            norm_dic['md_rmse_omb'] = []; norm_dic['ci_upper_rmse_omb'] = []; norm_dic['ci_lower_rmse_omb'] = [];
            norm_dic['md_std_omb'] = []; norm_dic['ci_upper_std_omb'] = []; norm_dic['ci_lower_std_omb'] = [];
            norm_dic['md_rmse_omb_nobc'] = []; norm_dic['ci_upper_rmse_omb_nobc'] = []; norm_dic['ci_lower_rmse_omb_nobc'] = [];
            norm_dic['md_std_omb_nobc'] = []; norm_dic['ci_upper_std_omb_nobc'] = []; norm_dic['ci_lower_std_omb_nobc'] = [];
            
            norm_dic['count_oma'] = [];
            norm_dic['md_rmse_oma'] = []; norm_dic['ci_upper_rmse_oma'] = []; norm_dic['ci_lower_rmse_oma'] = [];
            norm_dic['md_std_oma'] = []; norm_dic['ci_upper_std_oma'] = []; norm_dic['ci_lower_std_oma'] = [];
            norm_dic['md_rmse_oma_nobc'] = []; norm_dic['ci_upper_rmse_oma_nobc'] = []; norm_dic['ci_lower_rmse_oma_nobc'] = [];
            norm_dic['md_std_oma_nobc'] = []; norm_dic['ci_upper_std_oma_nobc'] = []; norm_dic['ci_lower_std_oma_nobc'] = [];
            
            for ivar in range(num_var):
                for ilev, lev in enumerate(levels):
                    for iexp in range(num_exp-1):
                        norm_dic['exp_ref'].append(exp_labels[0])
                        norm_dic['exp_tag'].append(exp_labels[iexp+1])
                        norm_dic['domain'].append(dom)
                        norm_dic['var_name'].append(tags[ivar])
                        norm_dic[f'{bin_label}'].append(lev)    
                        norm_dic['count_omb'].append(norm_results['count_omb_bin'][iexp, idom, ilev, ivar])
                        norm_dic['md_rmse_omb'].append(norm_results['md_rmse_omb_bin'][iexp, idom, ilev, ivar])
                        norm_dic['ci_upper_rmse_omb'].append(norm_results['ci_upper_rmse_omb_bin'][iexp, idom, ilev, ivar])
                        norm_dic['ci_lower_rmse_omb'].append(norm_results['ci_lower_rmse_omb_bin'][iexp, idom, ilev, ivar])
                        norm_dic['md_std_omb'].append(norm_results['md_std_omb_bin'][iexp, idom, ilev, ivar])
                        norm_dic['ci_upper_std_omb'].append(norm_results['ci_upper_std_omb_bin'][iexp, idom, ilev, ivar])
                        norm_dic['ci_lower_std_omb'].append(norm_results['ci_lower_std_omb_bin'][iexp, idom, ilev, ivar])
                        norm_dic['md_rmse_omb_nobc'].append(norm_results['md_rmse_omb_nobc_bin'][iexp, idom, ilev, ivar])
                        norm_dic['ci_upper_rmse_omb_nobc'].append(norm_results['ci_upper_rmse_omb_nobc_bin'][iexp, idom, ilev, ivar])
                        norm_dic['ci_lower_rmse_omb_nobc'].append(norm_results['ci_lower_rmse_omb_nobc_bin'][iexp, idom, ilev, ivar])
                        norm_dic['md_std_omb_nobc'].append(norm_results['md_std_omb_nobc_bin'][iexp, idom, ilev, ivar])
                        norm_dic['ci_upper_std_omb_nobc'].append(norm_results['ci_upper_std_omb_nobc_bin'][iexp, idom, ilev, ivar])
                        norm_dic['ci_lower_std_omb_nobc'].append(norm_results['ci_lower_std_omb_nobc_bin'][iexp, idom, ilev, ivar])
                        
                        norm_dic['count_oma'].append(norm_results['count_oma_bin'][iexp, idom, ilev, ivar])
                        norm_dic['md_rmse_oma'].append(norm_results['md_rmse_oma_bin'][iexp, idom, ilev, ivar])
                        norm_dic['ci_upper_rmse_oma'].append(norm_results['ci_upper_rmse_oma_bin'][iexp, idom, ilev, ivar])
                        norm_dic['ci_lower_rmse_oma'].append(norm_results['ci_lower_rmse_oma_bin'][iexp, idom, ilev, ivar])
                        norm_dic['md_std_oma'].append(norm_results['md_std_oma_bin'][iexp, idom, ilev, ivar])
                        norm_dic['ci_upper_std_oma'].append(norm_results['ci_upper_std_oma_bin'][iexp, idom, ilev, ivar])
                        norm_dic['ci_lower_std_oma'].append(norm_results['ci_lower_std_oma_bin'][iexp, idom, ilev, ivar])
                        norm_dic['md_rmse_oma_nobc'].append(norm_results['md_rmse_oma_nobc_bin'][iexp, idom, ilev, ivar])
                        norm_dic['ci_upper_rmse_oma_nobc'].append(norm_results['ci_upper_rmse_oma_nobc_bin'][iexp, idom, ilev, ivar])
                        norm_dic['ci_lower_rmse_oma_nobc'].append(norm_results['ci_lower_rmse_oma_nobc_bin'][iexp, idom, ilev, ivar])
                        norm_dic['md_std_oma_nobc'].append(norm_results['md_std_oma_nobc_bin'][iexp, idom, ilev, ivar])
                        norm_dic['ci_upper_std_oma_nobc'].append(norm_results['ci_upper_std_oma_nobc_bin'][iexp, idom, ilev, ivar])
                        norm_dic['ci_lower_std_oma_nobc'].append(norm_results['ci_lower_std_oma_nobc_bin'][iexp, idom, ilev, ivar])

            # save into .csv file (moved inside the domain loop -- see note above)
            filename = f'{csv_output}/{diag_obs}_{dom}_binned.csv'

            df = pd.DataFrame(norm_dic)
            df.to_csv(filename, index=False)

    return

def compile_results_and_save_into_csv(config, diag_obs, bin_method, results):

    csv_output = config['paths']['output_dir']+'/csv/'+diag_obs
    os.makedirs(csv_output, exist_ok=True)

    exp_labels = config['experiments']['labels']
    tags = odc.get_obs_config(diag_obs).tags

    # derive dimensions from `results` instead of relying on module globals
    num_var = len(results['variables'])
    levels = results['levels']

    ## unbinned variables:
    for iexp, exp in enumerate(exp_labels):
        dic={};
        dic['domain'] = []; dic['var_name'] = []; 
        dic['count_omb'] = [];  
        dic['mean_omb'] = []; dic['mean_omb_nobc'] = [];
        dic['rmse_omb'] = []; dic['rmse_omb_nobc'] = [];
        dic['std_omb'] = []; dic['std_omb_nobc'] = [];
        dic['spread_omb'] = []; 
        dic['totSpread_omb'] = [];
        
        dic['count_oma'] = [];  
        dic['mean_oma'] = []; dic['mean_oma_nobc'] = [];
        dic['rmse_oma'] = []; dic['rmse_oma_nobc'] = [];
        dic['std_oma'] = []; dic['std_oma_nobc'] = [];
        dic['spread_oma'] = []; 
        dic['totSpread_oma'] = [];
        
        for idom, dom in enumerate(config['plot_settings']['apply_domains']):
            dom_id = config['plot_settings']['apply_domains'][idom]
        
            for ivar in range(num_var):
                dic['domain'].append(dom)
                dic['var_name'].append(tags[ivar])
                dic['count_omb'].append(np.nanmean(results['count_omb'][iexp, idom, : ,ivar]))
                dic['mean_omb'].append(np.nanmean(results['mean_omb'][iexp, idom, : ,ivar]))
                dic['mean_omb_nobc'].append(np.nanmean(results['mean_omb_nobc'][iexp, idom, : ,ivar]))
                dic['rmse_omb'].append(np.nanmean(results['rmse_omb'][iexp, idom, : ,ivar]))
                dic['rmse_omb_nobc'].append(np.nanmean(results['rmse_omb_nobc'][iexp, idom, : ,ivar]))
                dic['std_omb'].append(np.nanmean(results['std_omb'][iexp, idom, : ,ivar]))
                dic['std_omb_nobc'].append(np.nanmean(results['std_omb_nobc'][iexp, idom, : ,ivar]))
                dic['spread_omb'].append(np.nanmean(results['spread_omb'][iexp, idom, : ,ivar]))
                dic['totSpread_omb'].append(np.nanmean(results['totSpread_omb'][iexp, idom, : ,ivar]))
    
                dic['count_oma'].append(np.nanmean(results['count_oma'][iexp, idom, : ,ivar]))
                dic['mean_oma'].append(np.nanmean(results['mean_oma'][iexp, idom, : ,ivar]))
                dic['mean_oma_nobc'].append(np.nanmean(results['mean_oma_nobc'][iexp, idom, : ,ivar]))
                dic['rmse_oma'].append(np.nanmean(results['rmse_oma'][iexp, idom, : ,ivar]))
                dic['rmse_oma_nobc'].append(np.nanmean(results['rmse_oma_nobc'][iexp, idom, : ,ivar]))
                dic['std_oma'].append(np.nanmean(results['std_oma'][iexp, idom, : ,ivar]))
                dic['std_oma_nobc'].append(np.nanmean(results['std_oma_nobc'][iexp, idom, : ,ivar]))
                dic['spread_oma'].append(np.nanmean(results['spread_oma'][iexp, idom, : ,ivar]))
                dic['totSpread_oma'].append(np.nanmean(results['totSpread_oma'][iexp, idom, : ,ivar]))
                
        # save into .csv file
        filename = f'{csv_output}/{diag_obs}_unbinned_{exp}.csv'  # contains all apply_domains, not just the last one
    
        df = pd.DataFrame(dic)
        df.to_csv(filename, index=False)

    ## binned variables:
    if not diag_obs.startswith("sfc"):
        bin_label = {1: "pressure", 2: "height", 3: "latitude"}.get(bin_method, "level")
    
        for iexp, exp in enumerate(exp_labels):
            dic={};
            dic['domain'] = []; dic[bin_label] = []; dic['var_name'] = []; 
            dic['count_omb'] = [];  
            dic['mean_omb'] = []; dic['mean_omb_nobc'] = [];
            dic['rmse_omb'] = []; dic['rmse_omb_nobc'] = [];
            dic['std_omb'] = []; dic['std_omb_nobc'] = [];
            dic['spread_omb'] = []; 
            dic['totSpread_omb'] = [];
            
            dic['count_oma'] = [];  
            dic['mean_oma'] = []; dic['mean_oma_nobc'] = [];
            dic['rmse_oma'] = []; dic['rmse_oma_nobc'] = [];
            dic['std_oma'] = []; dic['std_oma_nobc'] = [];
            dic['spread_oma'] = []; 
            dic['totSpread_oma'] = [];
            
            for idom, dom in enumerate(config['plot_settings']['apply_domains']):
                dom_id = config['plot_settings']['apply_domains'][idom]

                for ivar in range(num_var):
                    for ilev, lev in enumerate(levels):
                        # NOTE: use the outer `iexp` (this experiment) here -- this used to
                        # shadow it with `for iexp in range(num_exp-1):`, which silently
                        # replaced every output file's data with experiments 0..num_exp-2
                        # regardless of which experiment the file was named after, and
                        # dropped the last experiment's binned data entirely.
                        dic['domain'].append(dom)
                        dic['var_name'].append(tags[ivar])
                        dic[f'{bin_label}'].append(lev)
                        dic['count_omb'].append(np.nanmean(results['count_omb_bin'][iexp, idom, :, ilev, ivar]))
                        dic['mean_omb'].append(np.nanmean(results['mean_omb_bin'][iexp, idom, :, ilev, ivar]))
                        dic['mean_omb_nobc'].append(np.nanmean(results['mean_omb_nobc_bin'][iexp, idom, :, ilev, ivar]))
                        dic['rmse_omb'].append(np.nanmean(results['rmse_omb_bin'][iexp, idom, :, ilev, ivar]))
                        dic['rmse_omb_nobc'].append(np.nanmean(results['rmse_omb_nobc_bin'][iexp, idom, :, ilev, ivar]))
                        dic['std_omb'].append(np.nanmean(results['std_omb_bin'][iexp, idom, :, ilev, ivar]))
                        dic['std_omb_nobc'].append(np.nanmean(results['std_omb_nobc_bin'][iexp, idom, :, ilev, ivar]))
                        dic['spread_omb'].append(np.nanmean(results['spread_omb_bin'][iexp, idom, :, ilev, ivar]))
                        dic['totSpread_omb'].append(np.nanmean(results['totSpread_omb_bin'][iexp, idom, :, ilev, ivar]))

                        dic['count_oma'].append(np.nanmean(results['count_oma_bin'][iexp, idom, :, ilev, ivar]))
                        dic['mean_oma'].append(np.nanmean(results['mean_oma_bin'][iexp, idom, :, ilev, ivar]))
                        dic['mean_oma_nobc'].append(np.nanmean(results['mean_oma_nobc_bin'][iexp, idom, :, ilev, ivar]))
                        dic['rmse_oma'].append(np.nanmean(results['rmse_oma_bin'][iexp, idom, :, ilev, ivar]))
                        dic['rmse_oma_nobc'].append(np.nanmean(results['rmse_oma_nobc_bin'][iexp, idom, :, ilev, ivar]))
                        dic['std_oma'].append(np.nanmean(results['std_oma_bin'][iexp, idom, :, ilev, ivar]))
                        dic['std_oma_nobc'].append(np.nanmean(results['std_oma_nobc_bin'][iexp, idom, :, ilev, ivar]))
                        dic['spread_oma'].append(np.nanmean(results['spread_oma_bin'][iexp, idom, :, ilev, ivar]))
                        dic['totSpread_oma'].append(np.nanmean(results['totSpread_oma_bin'][iexp, idom, :, ilev, ivar]))

            # save into .csv file:
            filename = f'{csv_output}/{diag_obs}_binned_{exp}.csv'  # contains all apply_domains, not just the last one
            
            df = pd.DataFrame(dic)
            df.to_csv(filename, index=False)
        
    ## timeseries:
    for iexp, exp in enumerate(exp_labels):
        dic={};
        dic['domain'] = []; dic['var_name'] = []; dic['time']=[];
        dic['count_omb'] = [];  
        dic['mean_omb'] = []; dic['mean_omb_nobc'] = [];
        dic['rmse_omb'] = []; dic['rmse_omb_nobc'] = [];
        dic['std_omb'] = []; dic['std_omb_nobc'] = [];
        dic['spread_omb'] = []; 
        dic['totSpread_omb'] = [];
        
        dic['count_oma'] = [];  
        dic['mean_oma'] = []; dic['mean_oma_nobc'] = [];
        dic['rmse_oma'] = []; dic['rmse_oma_nobc'] = [];
        dic['std_oma'] = []; dic['std_oma_nobc'] = [];
        dic['spread_oma'] = []; 
        dic['totSpread_oma'] = [];
    
        for idom, dom in enumerate(config['plot_settings']['apply_domains']):
            dom_id = config['plot_settings']['apply_domains'][idom]
        
            for ivar in range(num_var):
                for itime, time in enumerate(results['times']):
                    dic['domain'].append(dom)
                    dic['var_name'].append(tags[ivar])
                    dic['time'].append(time)
                    dic['count_omb'].append(results['count_omb'][iexp, idom, itime ,ivar])
                    dic['mean_omb'].append(results['mean_omb'][iexp, idom, itime ,ivar])
                    dic['mean_omb_nobc'].append(results['mean_omb_nobc'][iexp, idom, itime ,ivar])
                    dic['rmse_omb'].append(results['rmse_omb'][iexp, idom, itime ,ivar])
                    dic['rmse_omb_nobc'].append(results['rmse_omb_nobc'][iexp, idom, itime ,ivar])
                    dic['std_omb'].append(results['std_omb'][iexp, idom, itime ,ivar])
                    dic['std_omb_nobc'].append(results['std_omb_nobc'][iexp, idom, itime ,ivar])
                    dic['spread_omb'].append(results['spread_omb'][iexp, idom, itime ,ivar])
                    dic['totSpread_omb'].append(results['totSpread_omb'][iexp, idom, itime ,ivar])
            
                    dic['count_oma'].append(results['count_oma'][iexp, idom, itime ,ivar])
                    dic['mean_oma'].append(results['mean_oma'][iexp, idom, itime ,ivar])
                    dic['mean_oma_nobc'].append(results['mean_oma_nobc'][iexp, idom, itime ,ivar])
                    dic['rmse_oma'].append(results['rmse_oma'][iexp, idom, itime ,ivar])
                    dic['rmse_oma_nobc'].append(results['rmse_oma_nobc'][iexp, idom, itime ,ivar])
                    dic['std_oma'].append(results['std_oma'][iexp, idom, itime ,ivar])
                    dic['std_oma_nobc'].append(results['std_oma_nobc'][iexp, idom, itime ,ivar])
                    dic['spread_oma'].append(results['spread_oma'][iexp, idom, itime ,ivar])
                    dic['totSpread_oma'].append(results['totSpread_oma'][iexp, idom, itime ,ivar])
        # save into .csv file
        filename = f'{csv_output}/{diag_obs}_timeseries_{exp}.csv'  # contains all apply_domains, not just the last one
        
        df = pd.DataFrame(dic)
        df.to_csv(filename, index=False)

    return




