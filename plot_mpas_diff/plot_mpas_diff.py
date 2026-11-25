#!/usr/bin/env python3
"""
Plot spatial differences of MPAS fields between files (e.g. an - bg, RTTOV - CRTM)
on the unstructured MPAS mesh, looping over cycles. Everything is controlled by a YAML config.

Usage:
  python plot_mpas_diff.py -c plot_mpas_diff.yaml
"""
import argparse
import math
import os
import re
from datetime import datetime, timedelta

import numpy as np
import netCDF4 as nc
import yaml
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

G = 9.80665
CYCLE_FMT = '%Y%m%d%H'


# ---- 設定檔 ----
def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)

    files = {}
    for key, ent in cfg['files'].items():
        if isinstance(ent, str):
            ent = {'path': ent}
        files[key] = {'path': ent['path'], 'label': ent.get('label', key)}
    cfg['files'] = files            # 路徑樣板，每個 cycle 再代入
    cfg['templates'] = cfg.get('templates') or {}
    cfg['cycles'] = build_cycles(cfg['cycles'])

    cfg['panels'] = [parse_panel(p, files) for p in cfg['panels']]
    default_levels = cfg.get('levels', [])
    cfg['variables'] = [parse_variable(v, default_levels) for v in cfg['variables']]

    out = cfg.setdefault('output', {})
    out.setdefault('dir', './figs/{cycle}')
    out['prefix'] = str(out.get('prefix', ''))
    return cfg


def build_cycles(entry):
    """cycles 可以是 list ['2024091612', ...]，或 {start, end, interval_hours}"""
    if isinstance(entry, (list, tuple)):
        return [datetime.strptime(str(c), CYCLE_FMT) for c in entry]
    start = datetime.strptime(str(entry['start']), CYCLE_FMT)
    end = datetime.strptime(str(entry.get('end', entry['start'])), CYCLE_FMT)
    step = timedelta(hours=entry.get('interval_hours', 1))
    cycles = []
    t = start
    while t <= end:
        cycles.append(t)
        t += step
    return cycles


def cycle_subs(templates, t):
    """內建 {cycle} (YYYYMMDDHH) 與 {date} (YYYY-MM-DD_HH.MM.SS)；templates 可引用它們"""
    subs = {'cycle': t.strftime(CYCLE_FMT), 'date': t.strftime('%Y-%m-%d_%H.%M.%S')}
    for k, v in templates.items():
        subs[k] = str(v).format(**subs)
    return subs


def parse_panel(entry, files):
    """panel 可以是 'a - b' 字串，或 {diff: 'a - b', title: '...'}"""
    if isinstance(entry, str):
        entry = {'diff': entry}
    m = re.fullmatch(r'\s*(\S+)\s+-\s+(\S+)\s*', entry['diff'])
    if m is None:
        raise ValueError(f"panel diff 格式須為 'fileA - fileB' (減號兩側要空白): {entry['diff']!r}")
    a, b = m.groups()
    for k in (a, b):
        if k not in files:
            raise KeyError(f'panel 使用了未定義的檔案 key: {k}')
    return {'a': a, 'b': b, 'title': entry.get('title')}


def parse_variable(entry, default_levels):
    """variable 可以是 'theta' 字串，或 {name: 'theta', levels: [...], vabs: 0.5}"""
    if isinstance(entry, str):
        entry = {'name': entry}
    name = entry['name']

    # 用 '+' 分隔，each term 一致要嘛都 _vi、要嘛都不是
    terms = [t.strip() for t in name.split('+')]
    vi_flags = [t.endswith('_vi') for t in terms]
    if len(set(vi_flags)) > 1:
        raise ValueError(f'{name}: 不能混用 _vi 與非 _vi')
    is_vi = vi_flags[0]
    base_vars = [t[:-3] if is_vi else t for t in terms]

    levels = None if is_vi else list(entry.get('levels', default_levels))
    if not is_vi and not levels:
        raise ValueError(f'{name}: 非 _vi 變數需要指定 levels')

    return {
        'name': name,
        'is_vi': is_vi,
        'base_vars': base_vars,
        # 混合比 (q 開頭) 才轉 g/kg；其他變數 (如 theta) 保留原單位
        'is_mixing_ratio': all(bv.startswith('q') for bv in base_vars),
        'tag': ' + '.join(base_vars) + (' (integrated)' if is_vi else ''),
        'levels': levels,
        'vabs': entry.get('vabs'),
    }


# ---- 垂直積分輔助函式 ----
def layer_dp(pressure_mid, surface_pressure):
    p_interface = 0.5 * (pressure_mid[:, :-1] + pressure_mid[:, 1:])
    p_bottom = surface_pressure[:, None]
    p_top = pressure_mid[:, -1:] - (p_interface[:, -1:] - pressure_mid[:, -1:])
    p_top = np.maximum(p_top, 0.0)
    p_bounds = np.concatenate([p_bottom, p_interface, p_top], axis=1)
    dp = -np.diff(p_bounds, axis=1)
    return dp


def vertical_integral(q_3d, dp):
    return np.nansum(q_3d * dp, axis=1) / G


# ---- 讀檔 ----
def read_array(ds, name, idx):
    return np.ma.filled(ds.variables[name][idx].astype(np.float64), np.nan)


def read_field(path, var, t):
    """
    回傳 (field, units)
      _vi    : (nCells,)          kg/m^2
      非 _vi : (nCells, nLevels)  只保留 var['levels'] 指定的層
    """
    ds = nc.Dataset(path)
    raw_3d = None
    for bv in var['base_vars']:
        v = read_array(ds, bv, (t, slice(None), slice(None)))
        raw_3d = v if raw_3d is None else raw_3d + v
    file_units = getattr(ds.variables[var['base_vars'][0]], 'units', '')

    if var['is_vi']:
        pbase_3d = read_array(ds, 'pressure_base', (t, slice(None), slice(None)))
        pp_3d = read_array(ds, 'pressure_p', (t, slice(None), slice(None)))
        psfc = read_array(ds, 'surface_pressure', (t, slice(None)))
        dp = layer_dp(pbase_3d + pp_3d, psfc)
        field = vertical_integral(raw_3d, dp)
        units = 'kg/m^2'
    else:
        nlev = raw_3d.shape[1]
        bad = [n for n in var['levels'] if not 0 <= n < nlev]
        if bad:
            raise IndexError(f"{var['name']}: levels {bad} 超出範圍 0~{nlev - 1}")
        field = raw_3d[:, var['levels']]
        if var['is_mixing_ratio']:
            field = field * 1e3
            units = 'g/kg'
        else:
            units = file_units
    ds.close()
    print(f"  read {var['name']} from {path}")
    return field, units


def read_latlon(path):
    ds = nc.Dataset(path)
    lat = np.degrees(np.asarray(ds.variables['latCell'][:]))
    lon = np.degrees(np.asarray(ds.variables['lonCell'][:]))
    ds.close()
    # latCell/lonCell 是 radian，轉成 degree，並把經度轉到 -180~180
    lon = np.where(lon > 180, lon - 360, lon)
    return lat, lon


def read_level_mean_pressure(path, t):
    ds = nc.Dataset(path)
    pbase = read_array(ds, 'pressure_base', (t, slice(None), slice(None)))
    ds.close()
    return np.nanmean(pbase, axis=0) / 100   # hPa


# ---- 繪圖 ----
def symmetric_range(diffs, vabs, auto_pct):
    if vabs is not None:
        return float(vabs), False
    all_abs = np.abs(np.concatenate([np.ravel(d) for d in diffs]))
    vabs = np.nanpercentile(all_abs, auto_pct)
    if not np.isfinite(vabs) or vabs == 0:
        vmax = np.nanmax(all_abs)
        vabs = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
    return float(vabs), True


def plot_figure(diffs, titles, lat, lon, vabs, suptitle, cbar_label, pcfg):
    npanel = len(diffs)
    ncols = pcfg.get('ncols') or npanel
    nrows = math.ceil(npanel / ncols)
    pw, ph = pcfg.get('panel_size', [6.7, 6.0])

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(pw * ncols, ph * nrows),
        subplot_kw={'projection': ccrs.PlateCarree()}, squeeze=False
    )
    axes = axes.ravel()
    extent = pcfg.get('extent') or [lon.min(), lon.max(), lat.min(), lat.max()]

    scs = []
    for ax, data, title in zip(axes, diffs, titles):
        sc = ax.scatter(
            lon, lat, c=data, s=pcfg.get('marker_size', 1), cmap=pcfg.get('cmap', 'RdBu_r'),
            vmin=-vabs, vmax=vabs, transform=ccrs.PlateCarree()
        )
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.set_title(title)
        scs.append(sc)
    for ax in axes[npanel:]:
        ax.set_visible(False)

    fig.suptitle(suptitle)
    # 所有 panel 共用一個 colorbar
    fig.colorbar(
        scs[0], ax=axes[:npanel].tolist(), orientation='horizontal',
        fraction=0.05, pad=0.08, extend='both', label=cbar_label
    )
    return fig


def save_figure(fig, out_path_noext, ocfg):
    formats = ocfg.get('format', 'png')
    if isinstance(formats, str):
        formats = [formats]
    for fmt in formats:
        path = f'{out_path_noext}.{fmt}'
        fig.savefig(path, dpi=ocfg.get('dpi', 200), bbox_inches='tight')
        print(f'  saved {path}')


def safe_name(s):
    return re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_')


def plot_variable(var, files, used_keys, panels, lat, lon, level_p, cycle_str,
                  out_dir, prefix, cfg):
    pcfg = cfg.get('plot') or {}
    ocfg = cfg['output']
    auto_pct = pcfg.get('auto_pct', 99)
    t = cfg.get('time_idx', 0)

    print(f"==> {cycle_str} {var['name']}")
    fields, units = {}, ''
    for k in used_keys:
        fields[k], units = read_field(files[k]['path'], var, t)

    titles = [
        p['title'] or f"{files[p['a']]['label']} - {files[p['b']]['label']} ({var['tag']})"
        for p in panels
    ]
    cbar_label = f"{var['name']} diff ({units})"

    if var['is_vi']:
        jobs = [(None, f"{var['name']} vertically integrated", f"{prefix}{safe_name(var['name'])}")]
    else:
        jobs = [
            (i, f"{var['name']} @ level {n} ~ {level_p[n]:.1f} hPa",
             f"{prefix}{safe_name(var['name'])}_lev{n:02d}")
            for i, n in enumerate(var['levels'])
        ]

    for i, level_label, fname in jobs:
        pick = (lambda f: f) if i is None else (lambda f: f[:, i])
        diffs = [pick(fields[p['a']]) - pick(fields[p['b']]) for p in panels]
        vabs, is_auto = symmetric_range(diffs, var['vabs'], auto_pct)
        print(f"  {level_label}: vabs = {vabs:.4g}" + (f' (auto, {auto_pct}th pct of |diff|)' if is_auto else ''))

        fig = plot_figure(diffs, titles, lat, lon, vabs, f'{cycle_str}  {level_label}', cbar_label, pcfg)
        save_figure(fig, os.path.join(out_dir, fname), ocfg)
        if not ocfg.get('show', False):
            plt.close(fig)


# ---- 主程式 ----
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-c', '--config', default='plot_mpas_diff.yaml', help='YAML 設定檔')
    args = parser.parse_args()

    cfg = load_config(args.config)
    panels, ocfg = cfg['panels'], cfg['output']
    t = cfg.get('time_idx', 0)
    skip_missing = cfg.get('skip_missing', True)

    show = ocfg.get('show', False)
    if not show:
        plt.switch_backend('Agg')

    # 只讀 panels 有用到的檔案
    used_keys = list(dict.fromkeys(k for p in panels for k in (p['a'], p['b'])))
    need_levels = any(not v['is_vi'] for v in cfg['variables'])

    first_subs = cycle_subs(cfg['templates'], cfg['cycles'][0])
    lat, lon = read_latlon(cfg['invariant'].format(**first_subs))
    level_p = None
    skipped = []

    for cycle in cfg['cycles']:
        subs = cycle_subs(cfg['templates'], cycle)
        files = {k: {'path': f['path'].format(**subs), 'label': f['label'].format(**subs)}
                 for k, f in cfg['files'].items()}

        missing = [files[k]['path'] for k in used_keys if not os.path.exists(files[k]['path'])]
        if missing:
            if not skip_missing:
                raise FileNotFoundError('\n'.join(missing))
            print(f"!! skip {subs['cycle']}: missing\n   " + '\n   '.join(missing))
            skipped.append(subs['cycle'])
            continue

        out_dir = ocfg['dir'].format(**subs)
        os.makedirs(out_dir, exist_ok=True)
        prefix = ocfg['prefix'].format(**subs)
        prefix = f'{prefix}_' if prefix else ''

        # 各層平均氣壓只用來標示層的位置，讀第一個可用 cycle 即可
        if need_levels and level_p is None:
            level_p = read_level_mean_pressure(files[used_keys[0]]['path'], t)

        for var in cfg['variables']:
            plot_variable(var, files, used_keys, panels, lat, lon, level_p, subs['cycle'],
                          out_dir, prefix, cfg)

    print(f"done: {len(cfg['cycles']) - len(skipped)}/{len(cfg['cycles'])} cycles plotted")
    if skipped:
        print(f"skipped cycles: {' '.join(skipped)}")

    if show:
        plt.show()


if __name__ == '__main__':
    main()
