#!/usr/bin/env python3
# main driver for running omb/oma rmse/std diagnostics

import obs_diag_config as odc
import obs_diag_stats as ods
import obs_diag_plots as odp
import warnings, os, shutil, sys
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union

warnings.filterwarnings("ignore")

def main(config_path: Optional[str] = None) -> None:
    """Main function to process data and generate plots"""
    config = odc.load_config(config_path)
    if config["mode"] not in odc.VALID_MODES:
        raise ValueError(f"unknown mode: {config['mode']!r} (expected one of {odc.VALID_MODES})")

    # Create output directories
    output_dir = config["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=False)
    
    # Copy the configuration file to the output folder for record
    # (skipped when running with the built-in DEFAULT_CONFIG, i.e. no yaml given):
    if config_path is not None:
        shutil.copy(config_path, output_dir)
    
    # Time control:
    start_time = datetime.strptime(config["time_settings"]["start_time"], "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(config["time_settings"]["end_time"], "%Y-%m-%d %H:%M:%S")
    interval_hour = config["time_settings"]["interval"]
    
    # Loop over different observation types:
    for diag_obs in config["sensors"]:
    
        # Adjust interval for specific observation types
        if diag_obs == 'sondes':
            interval = timedelta(hours=12)
            if start_time.hour in [6, 18]:
                adjusted_start = start_time + timedelta(hours=6)
            else:
                adjusted_start = start_time
        else:
            interval = timedelta(hours=interval_hour)
            adjusted_start = start_time
    
        print(f"Processing {diag_obs} from {adjusted_start} to {end_time}")
        
        ### Process data for this observation type
        bin_method = odc.get_bin_method(diag_obs)
        
        # each experiments own stats:
        results = ods.process_experiment(config, diag_obs, adjusted_start, end_time, interval)
        ods.compile_results_and_save_into_csv(config, diag_obs, bin_method, results)  # compile results and save them into .csv
    
        # normalized difference between tests experiments and the reference
        norm_results = ods.normalized_exp_diff_stats(config, results, ci_method=config['stats_settings']['ci_method'])
        ods.compile_norm_results_and_save_into_csv(config, diag_obs, bin_method, norm_results)  # compile norm_results and save them into .csv
    
        print('finish calulating the stats/ saving into csv.')
        
        ### graphics:
        if config['plot_settings']['save_graphics'] == True:
    
            print('generating plots...')
            
            # generate directory:
            os.makedirs(f'{output_dir}/graph/{diag_obs}', exist_ok=True)
            
            # Generate plots from results
            odp.generate_plots(results, config, diag_obs)
    
            # Generate plots from norm_results:
            odp.generate_norm_diff_plots(config, diag_obs)


if __name__ == "__main__":
    # Allow configuration file to be passed as argument
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)
