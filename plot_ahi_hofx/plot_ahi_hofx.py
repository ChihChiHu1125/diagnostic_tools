#!/usr/bin/env python3
"""
Plot AHI brightness-temperature maps (model H(x) from several experiments + OBS)
for every cycle in a time range, and save one PNG per cycle.

Usage:
    python plot_ahi_hofx.py plot_ahi_hofx.yaml
"""
import argparse
import os
import sys
from datetime import datetime, timedelta

import h5py
import numpy as np
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

TIME_FMT = '%Y%m%d%H'
MISSING_THRESHOLD = 1e10   # IODA fill value is ~ -3.3687953e+38

IR_VMIN, IR_VMAX = -110, 55


# ===== IR colormap (°C), sampled from the reference colorbar =====
def make_ir_cmap(vmin=IR_VMIN, vmax=IR_VMAX):
    nodes = [
        (-110,   '#423d41'),   # 深灰紫（最冷）
        (-100,   '#6c2165'),   # 暗紫
        ( -90,   '#9f019f'),   # 紫
        ( -80,   '#fe0102'),   # 紅
        ( -70,   '#fd7000'),   # 橘
        ( -60,   '#fff800'),   # 黃
        ( -55,   '#8aee00'),   # 黃綠
        ( -50.5, '#28dd01'),   # 綠
        ( -44.5, '#018e8e'),   # 藍綠
        ( -40,   '#0054ff'),   # 藍
        ( -30.6, '#0eadf7'),   # 淺藍
        ( -30.5, '#fbfffa'),   # 在約 -30°C 從淺藍直接跳到白色
        (  55,   '#000000'),   # 白 → 黑（線性灰階）
    ]
    pos = [(t - vmin) / (vmax - vmin) for t, _ in nodes]
    return LinearSegmentedColormap.from_list('ir_ref', list(zip(pos, [c for _, c in nodes])))


def time_list(cfg):
    t0 = datetime.strptime(str(cfg['start']), TIME_FMT)
    t1 = datetime.strptime(str(cfg['end']), TIME_FMT)
    dt = timedelta(hours=cfg.get('dt_hours', 1))
    times = []
    t = t0
    while t <= t1:
        times.append(t)
        t += dt
    return times


def read_bt(f, group, variable, idx):
    """Read one channel of group/variable in K, with missing values set to NaN."""
    v = f[group][variable][:, idx].astype('float64')
    v[~np.isfinite(v) | (np.abs(v) > MISSING_THRESHOLD)] = np.nan
    return v


def read_channel_number(f, idx):
    for path in ('MetaData/sensorChannelNumber', 'Channel'):
        if path in f:
            return int(f[path][idx])
    return None


def read_cycle(cfg, ctime):
    """Return lat, lon, list of (label, data °C), channel number for one cycle."""
    variable = cfg['variable']
    idx = cfg['channel_index']
    default_group = cfg.get('group', 'hofx')
    obs_group = cfg.get('obs_group')

    lat = lon = ch = None
    panels = []
    obs = None
    for exp in cfg['experiments']:
        fname = cfg['path_template'].format(root=cfg['root'], expt=exp['expt'], time=ctime)
        if not os.path.isfile(fname):
            raise FileNotFoundError(fname)
        with h5py.File(fname, 'r') as f:
            elat = f['MetaData']['latitude'][:]
            elon = f['MetaData']['longitude'][:]
            if lat is None:
                lat, lon = elat, elon
                ch = read_channel_number(f, idx)
                if obs_group:
                    obs = read_bt(f, obs_group, variable, idx)
            elif elat.shape != lat.shape or not (np.allclose(elat, lat) and np.allclose(elon, lon)):
                raise ValueError(f'locations in {fname} differ from the first experiment')
            group = exp.get('group', default_group)
            panels.append((exp['label'], read_bt(f, group, variable, idx) - 273.15))

    if obs is not None:
        panels.append((cfg.get('obs_label', 'OBS'), obs - 273.15))
    return lat, lon, panels, ch


def plot_map(ax, lon, lat, data, title, cmap, pcfg):
    lon_min, lon_max = pcfg['lon_range']
    lat_min, lat_max = pcfg['lat_range']
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    ax.coastlines(color='yellow', linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='yellow')

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, linestyle='--', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(range(-180, 181, 3))
    gl.ylocator = mticker.FixedLocator(range(-90, 91, 3))
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}

    ok = np.isfinite(data)
    sc = ax.scatter(lon[ok], lat[ok], c=data[ok], s=pcfg.get('marker_size', 5), marker='s',
                    linewidths=0, cmap=cmap, vmin=IR_VMIN, vmax=IR_VMAX,
                    transform=ccrs.PlateCarree())

    ax.set_title(title, fontsize=14)
    cb = plt.colorbar(sc, ax=ax, shrink=0.7)
    cb.set_ticks(range(-100, 41, 20))
    cb.set_label('°C')


def plot_cycle(cfg, t, cmap):
    ctime = t.strftime(TIME_FMT)
    lat, lon, panels, ch = read_cycle(cfg, ctime)

    pcfg = cfg.get('plot', {})
    ocfg = cfg['output']
    chtag = f'ch{ch}' if ch is not None else f'idx{cfg["channel_index"]}'

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(20 / 3 * n, 7),
                             subplot_kw={'projection': ccrs.PlateCarree()},
                             layout='constrained', squeeze=False)
    for ax, (label, data) in zip(axes[0], panels):
        plot_map(ax, lon, lat, data, f'{label} {chtag}', cmap, pcfg)

    tshow = t + timedelta(hours=pcfg.get('title_utc_offset_hours', 0))
    tz = pcfg.get('title_tz_label', 'UTC')
    fig.suptitle(f'{tshow:%Y-%m-%d %H:%M} {tz}', fontsize=18, fontweight='bold')

    os.makedirs(ocfg['dir'], exist_ok=True)
    out = os.path.join(ocfg['dir'], f'{ocfg.get("prefix", "ahi")}_{chtag}_{ctime}.png')
    fig.savefig(out, dpi=ocfg.get('dpi', 150))
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config', help='YAML configuration file')
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    skip_missing = cfg['output'].get('skip_missing', True)
    cmap = make_ir_cmap()

    for t in time_list(cfg['time']):
        try:
            out = plot_cycle(cfg, t, cmap)
            print(f'{t:%Y%m%d%H}: saved {out}')
        except FileNotFoundError as e:
            if not skip_missing:
                raise
            print(f'{t:%Y%m%d%H}: skipped, missing {e}', file=sys.stderr)


if __name__ == '__main__':
    main()
