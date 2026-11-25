#!/usr/bin/env python3
"""
每個 cycle 畫一張並列圖：
  左 / 中 : obs space 的 AHI 亮溫差 (例如 OmB、AmB)，來自 IODA obsout h5
  右       : model state 的 an - bg，畫在 MPAS 非結構網格上

圖數 = cycles x channels x (model 變數，非 _vi 每個 level 一張)。全部由 YAML 控制。

Usage:
  python plot_omb_amb.py -c plot_omb_amb.yaml
"""
import argparse
import os
import re
from datetime import datetime, timedelta

import h5py
import netCDF4 as nc
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

G = 9.80665
CYCLE_FMT = '%Y%m%d%H'
MISSING_THRESHOLD = 1e10   # IODA fill value ~ -3.3687953e+38


# =====================================================================
# 設定檔
# =====================================================================
def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)

    cfg['templates'] = cfg.get('templates') or {}
    cfg['cycles'] = build_cycles(cfg['cycles'])

    obs = cfg['obs']
    obs.setdefault('variable', 'brightnessTemperature')
    obs.setdefault('qc_group', None)
    obs.setdefault('qc_keep', [0])
    ch = obs.get('channel_index', 0)
    obs['channel_index'] = list(ch) if isinstance(ch, (list, tuple)) else [ch]
    obs['panels'] = [parse_obs_panel(p, obs) for p in obs['panels']]

    model = cfg['model']
    model['a'], model['b'] = parse_diff(model.get('diff', 'an - bg'), model['files'])
    model['files'] = {k: ({'path': v} if isinstance(v, str) else v) for k, v in model['files'].items()}
    for k, v in model['files'].items():
        v.setdefault('label', k)
    default_levels = model.get('levels', [])
    model['variables'] = [parse_variable(v, default_levels) for v in model['variables']]
    model.setdefault('time_idx', 0)

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
    cycles, t = [], start
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


def parse_diff(expr, keys=None):
    """'A - B' -> (A, B)；只有 'A' -> (A, None)。減號兩側要空白"""
    m = re.fullmatch(r'\s*(\S+)\s+-\s+(\S+)\s*', expr)
    if m:
        a, b = m.groups()
    elif re.fullmatch(r'\s*\S+\s*', expr):
        a, b = expr.strip(), None
    else:
        raise ValueError(f"格式須為 'A - B' 或 'A' (減號兩側要空白): {expr!r}")
    if keys is not None:
        for k in (a, b):
            if k is not None and k not in keys:
                raise KeyError(f'使用了未定義的 key: {k}')
    return a, b


def parse_obs_panel(entry, obs):
    """obs panel: {expr: 'ombg' | 'ombg - oman', title, vabs, qc_group}"""
    if isinstance(entry, str):
        entry = {'expr': entry}
    a, b = parse_diff(entry['expr'])
    return {
        'a': a, 'b': b,
        'expr': entry['expr'].strip(),
        'title': entry.get('title') or entry['expr'].strip(),
        'vabs': entry.get('vabs'),
        'qc_group': entry.get('qc_group', obs['qc_group']),
    }


def parse_variable(entry, default_levels):
    """model 變數：'theta' 或 {name: 'qc_vi + qr_vi', levels: [...], vabs: 0.1}"""
    if isinstance(entry, str):
        entry = {'name': entry}
    name = entry['name']
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
        'is_mixing_ratio': all(bv.startswith('q') for bv in base_vars),
        'tag': ' + '.join(base_vars) + (' (integrated)' if is_vi else ''),
        'levels': levels,
        'vabs': entry.get('vabs'),
    }


# =====================================================================
# 讀 obs (IODA h5)
# =====================================================================
def _need(f, group, var, path):
    if group not in f or var not in f[group]:
        groups = sorted(k for k in f.keys() if not re.search(r'_\d+$', k))
        raise KeyError(f'{path}: 找不到 {group}/{var}\n   可用的 group (已略去 member): {groups}')
    return f[group][var]


def read_obs_channel(f, group, var, idx, path):
    v = _need(f, group, var, path)[:, idx].astype(np.float64)
    v[~np.isfinite(v) | (np.abs(v) > MISSING_THRESHOLD)] = np.nan
    return v


def read_channel_number(f, idx):
    for p in ('MetaData/sensorChannelNumber', 'Channel'):
        if p in f:
            return int(f[p][idx])
    return None


def read_obs(path, obs):
    """回傳 lat, lon, {channel_index: {'ch': 通道號, 'panels': [data, ...]}}"""
    var, keep = obs['variable'], obs['qc_keep']
    out = {}
    with h5py.File(path, 'r') as f:
        lat = f['MetaData']['latitude'][:].astype(np.float64)
        lon = f['MetaData']['longitude'][:].astype(np.float64)
        lon = np.where(lon > 180, lon - 360, lon)
        for idx in obs['channel_index']:
            datas = []
            for p in obs['panels']:
                d = read_obs_channel(f, p['a'], var, idx, path)
                if p['b'] is not None:
                    d = d - read_obs_channel(f, p['b'], var, idx, path)
                if p['qc_group']:
                    qc = _need(f, p['qc_group'], var, path)[:, idx]
                    d[~np.isin(qc, keep)] = np.nan
                datas.append(d)
            out[idx] = {'ch': read_channel_number(f, idx), 'panels': datas}
    print(f'  read obs from {path}')
    return lat, lon, out


# =====================================================================
# 讀 model (MPAS)
# =====================================================================
def layer_dp(pressure_mid, surface_pressure):
    p_interface = 0.5 * (pressure_mid[:, :-1] + pressure_mid[:, 1:])
    p_bottom = surface_pressure[:, None]
    p_top = pressure_mid[:, -1:] - (p_interface[:, -1:] - pressure_mid[:, -1:])
    p_top = np.maximum(p_top, 0.0)
    p_bounds = np.concatenate([p_bottom, p_interface, p_top], axis=1)
    return -np.diff(p_bounds, axis=1)


def read_array(ds, name, idx):
    return np.ma.filled(ds.variables[name][idx].astype(np.float64), np.nan)


def read_model_field(path, var, t):
    """_vi: (nCells,) kg/m^2；非 _vi: (nCells, len(levels))"""
    ds = nc.Dataset(path)
    raw = None
    for bv in var['base_vars']:
        v = read_array(ds, bv, (t, slice(None), slice(None)))
        raw = v if raw is None else raw + v
    file_units = getattr(ds.variables[var['base_vars'][0]], 'units', '')
    if var['is_vi']:
        pmid = (read_array(ds, 'pressure_base', (t, slice(None), slice(None)))
                + read_array(ds, 'pressure_p', (t, slice(None), slice(None))))
        psfc = read_array(ds, 'surface_pressure', (t, slice(None)))
        field = np.nansum(raw * layer_dp(pmid, psfc), axis=1) / G
        units = 'kg/m^2'
    else:
        nlev = raw.shape[1]
        bad = [n for n in var['levels'] if not 0 <= n < nlev]
        if bad:
            raise IndexError(f"{var['name']}: levels {bad} 超出範圍 0~{nlev - 1}")
        field = raw[:, var['levels']]
        if var['is_mixing_ratio']:
            field, units = field * 1e3, 'g/kg'
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
    return lat, np.where(lon > 180, lon - 360, lon)


def read_level_mean_pressure(path, t):
    ds = nc.Dataset(path)
    pbase = read_array(ds, 'pressure_base', (t, slice(None), slice(None)))
    ds.close()
    return np.nanmean(pbase, axis=0) / 100   # hPa


# =====================================================================
# 繪圖
# =====================================================================
def symmetric_range(data, vabs, auto_pct):
    if vabs is not None:
        return float(vabs)
    a = np.abs(np.ravel(data))
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 1.0
    v = np.percentile(a, auto_pct)
    if not np.isfinite(v) or v == 0:
        v = a.max() if a.max() > 0 else 1.0
    return float(v)


def stats_text(d):
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 'N=0'
    return f'N={d.size}  mean={d.mean():.3f}  std={d.std():.3f}'


def draw_panel(ax, lon, lat, data, vabs, title, cbar_label, pcfg, extent, kind):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)

    gstep = pcfg.get('grid_step')
    if gstep:
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.4, linestyle='--', alpha=0.5)
        gl.top_labels = gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, gstep))
        gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, gstep))
        gl.xformatter, gl.yformatter = LongitudeFormatter(), LatitudeFormatter()

    ok = np.isfinite(data)
    if kind == 'obs':
        ms, marker = pcfg.get('obs_marker_size', 5), 's'
    else:
        ms, marker = pcfg.get('model_marker_size', 1), 'o'
    sc = ax.scatter(lon[ok], lat[ok], c=data[ok], s=ms, marker=marker, linewidths=0,
                    cmap=pcfg.get('cmap', 'RdBu_r'), vmin=-vabs, vmax=vabs,
                    transform=ccrs.PlateCarree())
    if pcfg.get('show_stats', True):
        title = f'{title}\n{stats_text(data)}'
    ax.set_title(title, fontsize=pcfg.get('title_size', 12))
    plt.colorbar(sc, ax=ax, orientation='horizontal', fraction=0.05, pad=0.06,
                 extend='both', label=cbar_label)


def make_figure(panels, suptitle, pcfg, extent):
    """panels: list of dict(lon, lat, data, vabs, title, cbar, kind)"""
    n = len(panels)
    pw, ph = pcfg.get('panel_size', [6.7, 6.5])
    fig, axes = plt.subplots(1, n, figsize=(pw * n, ph), layout='constrained',
                             subplot_kw={'projection': ccrs.PlateCarree()}, squeeze=False)
    for ax, p in zip(axes[0], panels):
        draw_panel(ax, p['lon'], p['lat'], p['data'], p['vabs'], p['title'], p['cbar'],
                   pcfg, extent, p['kind'])
    fig.suptitle(suptitle, fontsize=pcfg.get('suptitle_size', 15), fontweight='bold')
    return fig


def save_figure(fig, out_noext, ocfg):
    formats = ocfg.get('format', 'png')
    for fmt in ([formats] if isinstance(formats, str) else formats):
        path = f'{out_noext}.{fmt}'
        fig.savefig(path, dpi=ocfg.get('dpi', 200), bbox_inches='tight')
        print(f'  saved {path}')


def safe_name(s):
    return re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_')


# =====================================================================
# 一個 cycle
# =====================================================================
def plot_cycle(cfg, cycle, mlat, mlon, state):
    obs, model = cfg['obs'], cfg['model']
    pcfg, ocfg = cfg.get('plot') or {}, cfg['output']
    auto_pct = pcfg.get('auto_pct', 99)
    t = model['time_idx']

    subs = cycle_subs(cfg['templates'], cycle)
    obs_path = obs['path'].format(**subs)
    mfiles = {k: {'path': v['path'].format(**subs), 'label': v['label'].format(**subs)}
              for k, v in model['files'].items()}
    pa, pb = mfiles[model['a']], mfiles[model['b']]

    missing = [p for p in (obs_path, pa['path'], pb['path']) if not os.path.exists(p)]
    if missing:
        if not cfg.get('skip_missing', True):
            raise FileNotFoundError('\n'.join(missing))
        print(f"!! skip {subs['cycle']}: missing\n   " + '\n   '.join(missing))
        return False

    print(f"==> {subs['cycle']}")
    olat, olon, ochans = read_obs(obs_path, obs)

    if state['level_p'] is None and any(not v['is_vi'] for v in model['variables']):
        state['level_p'] = read_level_mean_pressure(pa['path'], t)

    extent = pcfg.get('extent') or [min(mlon.min(), olon.min()), max(mlon.max(), olon.max()),
                                    min(mlat.min(), olat.min()), max(mlat.max(), olat.max())]

    out_dir = ocfg['dir'].format(**subs)
    os.makedirs(out_dir, exist_ok=True)
    prefix = ocfg['prefix'].format(**subs)
    prefix = f'{prefix}_' if prefix else ''
    obs_units = obs.get('units', 'K')
    sensor = obs.get('label', 'AHI')

    # 先算 model 每個圖要的 (資料, 標題, 檔名片段)，與 channel 無關
    model_jobs = []
    for var in model['variables']:
        fa, units = read_model_field(pa['path'], var, t)
        fb, _ = read_model_field(pb['path'], var, t)
        diff = fa - fb
        mtitle = f"{pa['label']} - {pb['label']}"
        if var['is_vi']:
            model_jobs.append((diff, f"{mtitle}\n{var['tag']}", f"{var['name']} diff ({units})",
                               var['vabs'], safe_name(var['name'])))
        else:
            for i, n in enumerate(var['levels']):
                model_jobs.append((diff[:, i],
                                   f"{mtitle}\n{var['tag']} @ lev {n} ~ {state['level_p'][n]:.0f} hPa",
                                   f"{var['name']} diff ({units})", var['vabs'],
                                   f"{safe_name(var['name'])}_lev{n:02d}"))

    for idx, och in ochans.items():
        chtag = f"ch{och['ch']}" if och['ch'] is not None else f'idx{idx}'
        # 自動色階：各 panel 各自算，或 obs_common_range 時共用 (只算沒給 vabs 的 panel)
        common = None
        auto_data = [d for p, d in zip(obs['panels'], och['panels']) if p['vabs'] is None]
        if obs.get('obs_common_range', False) and auto_data:
            common = symmetric_range(np.concatenate([np.ravel(d) for d in auto_data]), None, auto_pct)
        obs_panels = []
        for p, d in zip(obs['panels'], och['panels']):
            if p['vabs'] is not None:
                vabs = float(p['vabs'])
            else:
                vabs = common if common is not None else symmetric_range(d, None, auto_pct)
            print(f"  {chtag} {p['title']}: ±{vabs:.4g}" + ('' if p['vabs'] is not None else ' (auto)'))
            obs_panels.append({'lon': olon, 'lat': olat, 'data': d,
                               'vabs': vabs,
                               'title': f"{sensor} {chtag}  {p['title']}",
                               'cbar': f"{p['expr']} ({obs_units})", 'kind': 'obs'})
        for mdata, mtitle, mcbar, mvabs, mtag in model_jobs:
            panels = obs_panels + [{'lon': mlon, 'lat': mlat, 'data': mdata,
                                    'vabs': symmetric_range(mdata, mvabs, auto_pct),
                                    'title': mtitle, 'cbar': mcbar, 'kind': 'model'}]
            fig = make_figure(panels, f"{subs['cycle']}  {sensor} {chtag}", pcfg, extent)
            save_figure(fig, os.path.join(out_dir, f'{prefix}{chtag}_{mtag}'), ocfg)
            if not ocfg.get('show', False):
                plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-c', '--config', default='plot_omb_amb.yaml', help='YAML 設定檔')
    args = parser.parse_args()

    cfg = load_config(args.config)
    show = cfg['output'].get('show', False)
    if not show:
        plt.switch_backend('Agg')

    first = cycle_subs(cfg['templates'], cfg['cycles'][0])
    mlat, mlon = read_latlon(cfg['model']['invariant'].format(**first))

    state = {'level_p': None}
    skipped = [c.strftime(CYCLE_FMT) for c in cfg['cycles']
               if not plot_cycle(cfg, c, mlat, mlon, state)]

    print(f"done: {len(cfg['cycles']) - len(skipped)}/{len(cfg['cycles'])} cycles plotted")
    if skipped:
        print(f"skipped cycles: {' '.join(skipped)}")
    if show:
        plt.show()


if __name__ == '__main__':
    main()
