#!/usr/bin/env python3

## Plot related subroutines
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union

import obs_diag_config as odc
from obs_diag_config import get_bin_method

# Preset colors
colors = [
'black', 'blue', 'red', 'green', 'orange', 'brown', 'cyan', 'purple',
'magenta', 'olive', 'teal', 'darkred', 'gold', 'navy', 'darkgreen'
]


def create_figure_layout(num_vars: int) -> Tuple[int, int]:
    """Determine optimal subplot layout based on number of variables"""
    if num_vars == 1:
        return 1, 1
    if num_vars <= 4:
        return 2, 2
    if num_vars <= 6:
        return 2, 3
    ncols = min(3, int(np.ceil(np.sqrt(num_vars))))
    nrows = int(np.ceil(num_vars / ncols))
    return nrows, ncols

def generate_plots(results: Dict, config: Dict, diag_obs: str) -> None:
    """Generate all plots from processed results"""

    '''
    # Preset colors
    colors = [
    'black', 'blue', 'red', 'green', 'orange', 'brown', 'cyan', 'purple',
    'magenta', 'olive', 'teal', 'darkred', 'gold', 'navy', 'darkgreen'
    ]
    '''
    
    # Create figure layout
    num_rows, num_cols = create_figure_layout(len(results['variables']))


    for idom, domain in enumerate(results['domain_names']):

        domain_suffix = f"_{domain}" if domain != "global" else ""

        # Time series plots
        generate_time_series_plots(results, num_rows, num_cols, colors, diag_obs, config, idom, domain, domain_suffix)

        # Vertical profile plots (sfc has no vertical levels -- skip this domain's
        # profile plot only, don't abort the remaining domains with `return`)
        if diag_obs.startswith("sfc"):
            continue

        radiance_data = diag_obs.startswith(("amsua", "mhs", "ahi", "abi", "iasi", "atms"))

        if (radiance_data and domain == "global") or (not radiance_data):
            generate_binned_profile_plots(results, num_rows, num_cols, colors, diag_obs, config, idom, domain, domain_suffix)

def generate_time_series_plots(results: Dict, num_rows: int, num_cols: int,
                                colors: List[str], diag_obs: str, config: Dict,
                                idom: int, domain_name: str, domain_suffix: str) -> None:
    """
    Generate time series plots with:
    - One figure per metric (RMSE, Bias, STD, Count)
    - Each figure contains panels for all variables
    - Uses existing plot_data_time function
    """
    # Configuration setup
    mode = config.get("mode", "da")  # run-wide da/hofx setting -- applies to all experiments
    exp_labels = config["experiments"].get("labels", [f"Exp{i+1}" for i in range(results['rmse_omb'].shape[0])])

    # figure output directory:
    graphic_output = config["paths"]["output_dir"]+f'/graph/{diag_obs}'
    os.makedirs(graphic_output, exist_ok=True)

    plot_oma = config["plot_settings"].get("plot_oma", True) and odc.has_oma(mode)
    suffix = "omboma" if plot_oma else "omb"

    # Define metrics to plot
    metrics = [
        ('rmse_omb', 'rmse_oma', 'RMSE'),
        ('mean_omb', 'mean_oma', 'Bias'),
        ('std_omb', 'std_oma', 'STD'),
        ('count_omb', 'count_oma', 'Count'),
        ('spread_omb', 'spread_oma', 'Spread'),
        ('totSpread_omb', 'totSpread_oma', 'totalSpread')
    ]

    # Create one figure per metric
    for metric_idx, (omb_key, oma_key, metric_name) in enumerate(metrics):
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols*13, num_rows*9))
        if num_rows * num_cols > 1:
            axs = axs.ravel()
        else:
            axs = [axs]

        # Plot each variable in its own panel
        for ivar, (var, unit, label) in enumerate(zip(results['variables'],
                                                    results['units'],
                                                    results['labels'])):
            ax = axs[ivar]
            domain_title = f" ({domain_name})" if domain_name != "global" else ""
            full_label = f"{label}{domain_title}"

            for iexp in range(len(exp_labels)):
                color = colors[iexp % len(colors)]
                exp_label = exp_labels[iexp]

                # Skip if OMB data is all NaN
                if np.all(np.isnan(results[omb_key][iexp, idom, :, ivar])):
                    continue

                # Plot OMB
                ylabel = f"{metric_name} ({unit})" if metric_name != 'Count' else 'Count'
                plot_data_time(ax, results['times'],
                             results[omb_key][iexp, idom, :, ivar],
                             label=f'{exp_label} - OMB',
                             diag_var=full_label,
                             var_units=ylabel,
                             linecolor=color,
                             linestyle='-')

               # Plot OMA if available
                if plot_oma and oma_key in results:
                    plot_data_time(ax, results['times'],
                                 results[oma_key][iexp, idom, :, ivar],
                                 label=f'{exp_label} - OMA',
                                 diag_var=full_label,
                                 var_units=ylabel,
                                 linecolor=color,
                                 linestyle='--')

            if ivar == len(results['variables']) - 1:
                ax.legend(fontsize=12)

        # Turn off unused axes
        for i in range(len(results['variables']), num_rows * num_cols):
            axs[i].axis('off')

        # Save figure
        plt.tight_layout()
        output_file = f"{graphic_output}/time_series_{metric_name.lower()}_{diag_obs}_{suffix}{domain_suffix}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

def plot_data_time(ax: plt.Axes, times: List[datetime], data: np.ndarray, label: str, diag_var: str,
    var_units: str, linecolor: str = 'black', linestyle: str = '-', linewidth: float = 2.5) -> None:

    # Calculate total time span in days
    total_days = (times[-1] - times[0]).total_seconds() / 86400

    # Determine label skipping interval based on total duration
    if total_days <= 5:
        skip_day = 1    # Show all labels for very short periods
    elif total_days <= 10:
        skip_day = 2     # Show every 2-day label for 1-week spans
    elif total_days <= 20:
        skip_day = 3     # Show every 3-day label for half-month spans
    elif total_days <= 50:
        skip_day = 5     # Show every 5-day label for 1-month spans
    elif total_days <= 90:
        skip_day = 7     # Show every week label for 3-month spans
    else:
        skip_day = 14    # Bi-weekly labeling for longer spans

    x_labels = []
    x_time = times[0]
    for itime in times:
        if (itime.day == 1 and itime.hour == 0) or itime == times[0]:
            x_labels.append(f'{itime.strftime("%HZ")} \n{itime.day} {itime.strftime("%b")}')
            x_time = itime
        elif itime.hour == 0 and (itime.day - x_time.day) % skip_day == 0:
            x_labels.append(f'{itime.strftime("%HZ")} \n{itime.day}')
        elif itime == times[-1]:
            x_labels.append(f'{itime.strftime("%HZ")} \n{itime.day}')
        else:
            x_labels.append('')  # Empty label for other times
    # Extend axis limits slightly
    padding = 0.5 * (times[1] - times[0])
    ax.set_xlim(times[0] - padding, times[-1] + padding)

    # Set ticks at all times, but show labels only at selected ones
    ax.set_xticks(times)
    ax.set_xticklabels(x_labels, ha='center')

    # Plot the data
    ax.plot(times, data, label=label, color=linecolor, linestyle=linestyle, linewidth=2.5)

    # Axis formatting
    ax.set_xlabel('Assimilation cycles', fontsize=22)
    ax.set_ylabel(f'{var_units}', fontsize=22)
    ax.set_title(diag_var, fontsize=24)
    ax.tick_params(axis='both', labelsize=18)
    ax.grid(True)

def generate_binned_profile_plots(results: Dict, num_rows: int, num_cols: int,
                                colors: List[str], diag_obs: str, config: Dict,
                                idom: int, domain_name: str, domain_suffix: str) -> None:
    """
    Generate vertical profile plots with:
    - One figure per metric (RMSE, Bias, STD, Count)
    - Each figure contains panels for all variables
    - Consistent with time series plot organization
    """
    # Configuration setup
    mode = config.get("mode", "da")  # run-wide da/hofx setting -- applies to all experiments
    exp_labels = config["experiments"].get("labels", [f"Exp{i+1}" for i in range(results['rmse_omb'].shape[0])])

    # figure output directory:
    graphic_output = config["paths"]["output_dir"]+f'/graph/{diag_obs}'
    os.makedirs(graphic_output, exist_ok=True)

    plot_oma = config["plot_settings"].get("plot_oma", False) and odc.has_oma(mode)
    suffix = "omboma" if plot_oma else "omb"
    bin_method = get_bin_method(diag_obs)

    # Determine bin label based on observation type
    bin_label = {1: "pressure", 2: "height", 3: "latitude"}.get(bin_method, "level")

    # Define metrics to plot
    metrics = [
        ('rmse_omb_bin', 'rmse_oma_bin', 'RMSE'),
        ('mean_omb_bin', 'mean_oma_bin', 'Bias'),
        ('std_omb_bin', 'std_oma_bin', 'STD'),
        ('count_omb_bin', 'count_oma_bin', 'Count'),
        ('spread_omb_bin', 'spread_oma_bin', 'Spread'),
        ('totSpread_omb_bin', 'totSpread_oma_bin', 'totalSpread')
    ]

    # Calculate time-averaged statistics for this domain
    stats = {}
    for omb_key, oma_key, _ in metrics:
        stats[omb_key] = np.nanmean(results[omb_key][:, idom, :, :, :], axis=1)
        if plot_oma and oma_key in results:
            stats[oma_key] = np.nanmean(results[oma_key][:, idom, :, :, :], axis=1)

    # Create one figure per metric
    for metric_idx, (omb_key, oma_key, metric_name) in enumerate(metrics):
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols*5, num_rows*6))
        if num_rows * num_cols > 1:
            axs = axs.ravel()
        else:
            axs = [axs]

        # Plot each variable in its own panel
        for ivar, (var, unit, label) in enumerate(zip(results['variables'],
                                                    results['units'],
                                                    results['labels'])):
            ax = axs[ivar]
            domain_title = f" ({domain_name})" if domain_name != "global" else ""
            full_label = f"{label}{domain_title}"

            for iexp in range(len(exp_labels)):
                color = colors[iexp % len(colors)]
                exp_label = exp_labels[iexp]

                # Skip if OMB data is all NaN
                if np.all(np.isnan(stats[omb_key][iexp, :, ivar])):
                    continue

                # Plot OMB
                ylabel = f"{metric_name} ({unit})" if metric_name != 'Count' else 'Count'
                plot_data_binned_profile(ax,
                                       stats[omb_key][iexp, :, ivar],
                                       results['levels'],
                                       f'{exp_label} - OMB',
                                       full_label,
                                       ylabel,
                                       color, '-')

                # Plot OMA if available
                if plot_oma and oma_key in stats:
                    plot_data_binned_profile(ax,
                                           stats[oma_key][iexp, :, ivar],
                                           results['levels'],
                                           f'{exp_label} - OMA',
                                           full_label,
                                           ylabel,
                                           color, '--')

            # Configure subplot
            if ivar == len(results['variables']) - 1:
                ax.legend(fontsize=12)

        # Turn off unused axes
        for i in range(len(results['variables']), num_rows * num_cols):
            axs[i].axis('off')

        # Save figure
        plt.tight_layout()
        output_file = f"{graphic_output}/binned_{bin_label}_{metric_name.lower()}_{diag_obs}_{suffix}{domain_suffix}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

def plot_data_binned_profile(ax: plt.Axes, data: np.ndarray, levels: np.ndarray,
                   label: str, diag_var: str, var_units: str,
                   linecolor: str = 'black', linestyle: str = '-',
                   counts: Optional[np.ndarray] = None) -> None:
    """Plot data against binned levels on the given axes"""
    level_indices = np.arange(len(levels))

    ax.plot(data, level_indices, label=label, color=linecolor,
            linestyle=linestyle, linewidth=2.5)

    # Configure y-axis
    ax.set_yticks(level_indices)
    ax.set_yticklabels([f'{lvl}' for lvl in levels])

    # Set axis labels based on binning method
    if levels[1] > levels[0]:
        ylabel = 'Latitude (°)' if 'Channel' in diag_var else 'Height (km)'
    else:
        ylabel = 'Pressure (hPa)'
        ax.invert_yaxis()

    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xlabel(var_units, fontsize=16)
    ax.set_title(diag_var, fontsize=20)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True)

    # Add counts if provided
    if counts is not None:
        if len(counts) != len(levels):
            raise ValueError("counts must have same length as levels")

        ymin, ymax = ax.get_ylim()
        for idx, count in zip(level_indices, counts):
            ax.text(1.02, idx, f"{int(count)}",
                   transform=ax.get_yaxis_transform(),
                   ha='left', va='center', fontsize=14)

        ax.text(1.15, (ymin + ymax)/2, 'Counts',
               transform=ax.get_yaxis_transform(),
               rotation=90, va='center', ha='center',
               fontsize=16)

def generate_norm_diff_plots(config, diag_obs):
    """ the driver script for plotting the normalized error (rmse/std) omb/oma plots """
    """ will call the subroutine plot_normdiff_errbar for actual plotting """
    
    # figure output directory:
    graphic_output = config["paths"]["output_dir"]+f'/graph/{diag_obs}/norm_diff/'
    os.makedirs(graphic_output, exist_ok=True)
    
    # Configuration setup
    bin_method     = get_bin_method(diag_obs)
    
    # Determine bin label based on observation type
    bin_label = {1: "pressure", 2: "height", 3: "latitude"}.get(bin_method, "level")

    # (omb/oma, rmse/std) combinations to plot; *_nobc only for radiances,
    # oma only when the run-wide mode has an analysis (i.e. "da")
    innov_types = ['omb', 'omb_nobc']
    if odc.has_oma(config['mode']):
        innov_types += ['oma', 'oma_nobc']
    combos = [(omb_oma, rmse_std)
              for omb_oma in innov_types
              for rmse_std in ('rmse', 'std')
              if bin_method == 3 or not omb_oma.endswith('_nobc')]
    
    # fig size configuration
    figsize_x = 5;
    figsize_y = 5;
    figsize = (figsize_x, figsize_y)

    csv_root = config['paths']['output_dir']+'/csv/'

    ### for unbinned variables (single plot)

    for idom, dom in enumerate(config['plot_settings']['apply_domains']): 
    
        df = pd.read_csv(f'{csv_root}/{diag_obs}/{diag_obs}_{dom}_unbinned.csv')
        
        domain_suffix = f"_{dom}" if dom != "global" else ""
        df_dom = df[df['domain'] == dom]
        
        for omb_oma, rmse_std in combos:
            if df_dom[f'md_{rmse_std}_{omb_oma}'].isna().all():  # e.g. *_nobc without ObsBias
                continue
            fig, ax = plt.subplots(figsize=figsize)
            plot_normdiff_errbar(ax, config, figsize, df_dom, omb_oma, rmse_std, 'var_name', ylabel='', title=f'{diag_obs} {domain_suffix}')
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))    
            plt.tight_layout()
            output_file = f"{graphic_output}/unbinned_{omb_oma}_{rmse_std}_{diag_obs}{domain_suffix}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

    ### for binned variables (each subplot -> one variable)
    if not diag_obs.startswith("sfc"):
        
        for idom, dom in enumerate(config['plot_settings']['apply_domains']): 
            
            df = pd.read_csv(f'{csv_root}/{diag_obs}/{diag_obs}_{dom}_binned.csv')
            
            domain_suffix = f"_{dom}" if dom != "global" else ""
            df_dom = df[df['domain'] == dom]
        
            for omb_oma, rmse_std in combos:
                if df_dom[f'md_{rmse_std}_{omb_oma}'].isna().all():
                    continue
                fig, axs = subplot_for_binned_variables(config, diag_obs, df_dom, omb_oma=omb_oma, rmse_std=rmse_std, bin_label=bin_label)
                output_file = f"{graphic_output}/binned_{omb_oma}_{rmse_std}_{diag_obs}{domain_suffix}.png"
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close(fig)
    return

def subplot_for_binned_variables(config, diag_obs, df_dom, omb_oma, rmse_std, bin_label):
    num_rows, num_cols = create_figure_layout(len(df_dom['var_name'].unique()));
    figsize_x = 4
    figsize_y = 5
    figsize = (figsize_x, figsize_y)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols*figsize_x, num_rows*figsize_y))
    if num_rows * num_cols > 1:
        axs = axs.ravel()
    else:
        axs = [axs]

    var_list = list(df_dom['var_name'].unique())
    
    for ivar, var in enumerate(var_list):
        ax = axs[ivar]
        df_dom_var = df_dom[df_dom['var_name']==var]
        plot_normdiff_errbar(ax, config, figsize, df_dom_var, omb_oma, rmse_std, bin_label, ylabel=bin_label, title=f'{diag_obs} [{var}]')
        plt.tight_layout()
        if ivar==0:
            ax.legend(fontsize=12)
        
    # Turn off unused axes
    for i in range(len(var_list), num_rows * num_cols):
        axs[i].axis('off')
            
    return fig, axs
    
def plot_normdiff_errbar(ax, config, figsize, df_dom, omb_oma, rmse_std, plt_yvar, ylabel, title):
    """ plot the normalized error (rmse/std) of omb/oma with confidence interval """
    """ the script for only one plot """

    if len(config['experiments']['labels']) < 2:
        # nothing to compare against a reference experiment
        return

    for iexp, exp in enumerate(config['experiments']['labels']):
        if iexp>0:
            df_dom_exp = df_dom[df_dom['exp_tag']==exp]
            
            #plt_yticks = df_dom_exp['var_name']
            plt_yticks = df_dom_exp[plt_yvar]
            plt_yaxis  = np.arange(len(plt_yticks))
        
            plt_omf = df_dom_exp[f'md_{rmse_std}_{omb_oma}']
    
            err_upper = df_dom_exp[f'ci_upper_{rmse_std}_{omb_oma}'] - plt_omf
            err_lower = plt_omf - df_dom_exp[f'ci_lower_{rmse_std}_{omb_oma}']
            
            xerr = np.vstack([err_lower, err_upper])
        
            color = colors[iexp % len(colors)]
            ax.errorbar(plt_omf, plt_yaxis, xerr=xerr, fmt='s', color=color, ecolor=color, \
                        capsize=0, linestyle='None', markersize=1.5)
            ax.plot(plt_omf, plt_yaxis, color=color, linewidth=2, label=f'{exp}')
        
    ax.axvline(100,linestyle='--', color='black')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_yticks(plt_yaxis, plt_yticks, fontsize=12)
    ax.set_xlabel(f'normalized {omb_oma} {rmse_std} (%)',fontsize=12)
    ax.set_ylabel(f'{ylabel}',fontsize=12)
    ax.grid(True)
    ax.set_title(f'{title} ', fontsize=13.5)