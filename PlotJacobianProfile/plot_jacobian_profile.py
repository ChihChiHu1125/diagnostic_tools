#!/usr/bin/env python3
"""
Jacobian vs. vertical profile from MPAS-JEDI HofX GOMsaver/YDIAGsaver output.

  python plot_jacobian_profile.py plot_jacobian_profile.yaml

Inputs per dataset (one HofX dbOut directory):
  obsout_<app>_<instrument>.h5          (or obsout_<app>_<instrument>_NNNN.h5 shards)
  geoval_<app>_<instrument>_NNNN.nc4    (or the concatenated geoval_<app>_<instrument>.nc4)
  ydiags_<app>_<instrument>_NNNN.nc4    (or the concatenated ydiags_<app>_<instrument>.nc4)

geoval/ydiags shards carry no lat/lon; like tools/concatenate.py of the workflow, they
are concatenated in rank (file name) order, which is the obsout Location order. The
script checks the location counts, and that clear model columns give
model BT == clear-sky BT (alignment_check).

Units in the figures:
  T  : K,    dBT/dT : K/K
  q* : g/kg, dBT/dq : K/(g/kg)
    CRTM  qv geoval is kg/kg, but its Jacobian is already K/(g/kg) (CRTM absorber unit);
    CRTM  hydrometeors are layer mass content m (kg/m^2): q = m g/dp, dBT/dq = dBT/dm dp/g;
    RTTOV q* geovals are kg/kg (moist air), Jacobians K/(kg/kg).
  dp comes from GeoVaLs air_pressure_levels (the same dp MPAS-JEDI used to build m).
"""

import argparse
import glob
import os
import sys
import warnings

import h5py
import numpy as np
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import ScaledTranslation

GRAVITY = 9.80665          # m/s^2, same as MPAS gravity constant
MISSING_ABS = 1.0e30       # |value| above this is missing (IODA -3.37e38, ydiags 9.97e36)

BT_CLEAR = 'brightness_temperature_assuming_clear_sky'
JAC_PREFIX = 'brightness_temperature_jacobian_'

# short name -> GeoVaLs name, Jacobian name (without channel), unit conversion kind
VARMAP = {
  'crtm': {
    'T':  ('air_temperature', 'air_temperature', 'T'),
    'qv': ('water_vapor_mixing_ratio_wrt_dry_air',
           'water_vapor_mixing_ratio_wrt_dry_air', 'qv_crtm'),
    'qc': ('mass_content_of_cloud_liquid_water_in_atmosphere_layer',
           'mass_content_of_cloud_liquid_water_in_atmosphere_layer', 'mass_crtm'),
    'qi': ('mass_content_of_cloud_ice_in_atmosphere_layer',
           'mass_content_of_cloud_ice_in_atmosphere_layer', 'mass_crtm'),
    'qr': ('mass_content_of_rain_in_atmosphere_layer',
           'mass_content_of_rain_in_atmosphere_layer', 'mass_crtm'),
    'qs': ('mass_content_of_snow_in_atmosphere_layer',
           'mass_content_of_snow_in_atmosphere_layer', 'mass_crtm'),
    'qg': ('mass_content_of_graupel_in_atmosphere_layer',
           'mass_content_of_graupel_in_atmosphere_layer', 'mass_crtm'),
  },
  'rttov': {
    'T':  ('air_temperature', 'air_temperature', 'T'),
    # RTTOV names its q Jacobian "wrt_dry_air", but it is d/d(q wrt moist air, kg/kg)
    'qv': ('water_vapor_mixing_ratio_wrt_moist_air',
           'water_vapor_mixing_ratio_wrt_dry_air', 'kgkg'),
    'qc': ('cloud_liquid_water_mixing_ratio_wrt_moist_air',
           'cloud_liquid_water_mixing_ratio_wrt_moist_air', 'kgkg'),
    'qi': ('cloud_ice_mixing_ratio_wrt_moist_air',
           'cloud_ice_mixing_ratio_wrt_moist_air', 'kgkg'),
    'qr': ('rain_water_mixing_ratio_wrt_moist_air',
           'rain_water_mixing_ratio_wrt_moist_air', 'kgkg'),
    'qs': ('snow_water_mixing_ratio_wrt_moist_air',
           'snow_water_mixing_ratio_wrt_moist_air', 'kgkg'),
    'qg': ('graupel_mixing_ratio_wrt_moist_air',
           'graupel_mixing_ratio_wrt_moist_air', 'kgkg'),
  },
}

LONG_NAME = {'T': 'temperature', 'qv': 'water vapor', 'qc': 'cloud water',
             'qi': 'cloud ice', 'qr': 'rain', 'qs': 'snow', 'qg': 'graupel'}


_WARNED = set()


def warn(msg):
  if msg in _WARNED:   # print each distinct warning once
    return
  _WARNED.add(msg)
  print('WARNING: ' + msg, flush=True)


def die(msg):
  sys.exit('ERROR: ' + msg)


def fmt(s, templates):
  return str(s).format(**templates) if isinstance(s, str) else s


def to_nan(a):
  a = np.asarray(a, dtype=np.float64)
  a[~np.isfinite(a) | (np.abs(a) > MISSING_ABS)] = np.nan
  return a


def find_files(directory, prefix, app, inst, ext):
  '''single (concatenated) file if present, else rank-ordered shards'''
  single = os.path.join(directory, '{}_{}_{}.{}'.format(prefix, app, inst, ext))
  if os.path.isfile(single):
    return [single]
  shards = sorted(glob.glob(os.path.join(
    directory, '{}_{}_{}_[0-9][0-9][0-9][0-9].{}'.format(prefix, app, inst, ext))))
  if not shards:
    die('no {}_{}_{} files ({}) in {}'.format(prefix, app, inst, ext, directory))
  return shards


def read_concat(files, name):
  '''concatenate one variable along the first (location) axis; None if absent'''
  parts = []
  for f in files:
    with h5py.File(f, 'r') as h:
      if name not in h:
        return None
      parts.append(h[name][...])
  return np.concatenate(parts, axis=0)


def count_locs(files, name):
  n = 0
  for f in files:
    with h5py.File(f, 'r') as h:
      n += h[name].shape[0]
  return n


class Dataset:
  def __init__(self, key, cfg, inst, app, templates):
    self.key = key
    self.operator = str(cfg.get('operator', 'crtm')).lower()
    if self.operator not in VARMAP:
      die('dataset {}: operator must be crtm or rttov'.format(key))
    self.label = cfg.get('label', key)
    self.color = cfg.get('color', None)
    self.linestyle = cfg.get('linestyle', '-')
    self.dir = fmt(cfg['dir'], templates)
    self.inst = inst
    self._geo = {}
    self._diag = {}

    obsFiles = find_files(self.dir, 'obsout', app, inst, 'h5')
    self.geoFiles = find_files(self.dir, 'geoval', app, inst, 'nc4')
    self.diagFiles = find_files(self.dir, 'ydiags', app, inst, 'nc4')

    with h5py.File(obsFiles[0], 'r') as h:
      self.channels = [int(c) for c in h['Channel'][...]]
    self.lat = to_nan(read_concat(obsFiles, 'MetaData/latitude'))
    self.lon = to_nan(read_concat(obsFiles, 'MetaData/longitude'))
    self.obs = to_nan(read_concat(obsFiles, 'ObsValue/brightnessTemperature'))
    self.hofx = to_nan(read_concat(obsFiles, 'hofx/brightnessTemperature'))
    self.qc = read_concat(obsFiles, 'EffectiveQC/brightnessTemperature')
    self.nlocs = self.lat.size

    # shard alignment: location counts must match obsout
    nGeo = count_locs(self.geoFiles, 'air_temperature')
    with h5py.File(self.diagFiles[0], 'r') as h:
      # skip the netCDF dimension datasets (<var>_nval, length = nval, not nlocs)
      anyDiag = [v for v in h.keys() if (v.startswith(BT_CLEAR) or v.startswith(JAC_PREFIX))
                 and not v.endswith('_nval')]
    if not anyDiag:
      die('{}: no ydiags variables in {}'.format(key, self.diagFiles[0]))
    nDiag = count_locs(self.diagFiles, anyDiag[0])
    if not (nGeo == nDiag == self.nlocs):
      die('{}: location counts differ: obsout {}, geoval {} ({} files), ydiags {} ({} files)'
          .format(key, self.nlocs, nGeo, len(self.geoFiles), nDiag, len(self.diagFiles)))
    print('{}: {} ({}), {} locations, {} geoval / {} ydiags files'.format(
      key, self.dir, self.operator, self.nlocs, len(self.geoFiles), len(self.diagFiles)))

  def chIndex(self, ch):
    if ch not in self.channels:
      die('{}: channel {} not in obsout channels {}'.format(self.key, ch, self.channels))
    return self.channels.index(ch)

  def geo(self, name):
    if name not in self._geo:
      a = read_concat(self.geoFiles, name)
      self._geo[name] = None if a is None else to_nan(a)
    return self._geo[name]

  def diag(self, name):
    if name not in self._diag:
      a = read_concat(self.diagFiles, name)
      self._diag[name] = None if a is None else to_nan(a)
    return self._diag[name]

  def bt(self, ch):
    return self.hofx[:, self.chIndex(ch)]

  def obsBT(self, ch):
    return self.obs[:, self.chIndex(ch)]

  def qcPass(self, ch):
    return self.qc[:, self.chIndex(ch)] == 0

  def clear(self, ch):
    a = self.diag('{}_{}'.format(BT_CLEAR, ch))
    if a is None:
      die('{}: {}_{} not in ydiags'.format(self.key, BT_CLEAR, ch))
    return a[:, 0]

  def pressure(self, how):
    '''(layer pressure [hPa], layer dp [Pa]), both (nlocs, nlayers)'''
    plev = self.geo('air_pressure_levels')
    if plev is None:
      die('{}: air_pressure_levels not in geovals'.format(self.key))
    dp = np.abs(np.diff(plev, axis=1))
    p = None
    if how in ('auto', 'air_pressure'):
      p = self.geo('air_pressure')
      if p is None and how == 'air_pressure':
        die('{}: air_pressure not in geovals'.format(self.key))
    if p is None:
      p = 0.5 * (plev[:, :-1] + plev[:, 1:])
    if np.nanmax(p) > 2000.0:   # Pa -> hPa
      p = p / 100.0
    return p, dp

  def profileAndJacobian(self, var, ch, how):
    '''converted (profile, jacobian, pressure hPa, units), or None if the profile itself is
       unavailable; a missing Jacobian (e.g. not in the operator's Jacobian list) gives an
       all-NaN Jacobian, so the profile/increment is still plotted and the Jacobian is blank'''
    vm = VARMAP[self.operator]
    if var not in vm:
      die('unknown variable {} (options: {})'.format(var, list(vm.keys())))
    geoName, jacName, kind = vm[var]
    prof = self.geo(geoName)
    if prof is None:
      warn('{}: {} not plotted: GeoVaLs {} not in geoval files'.format(self.key, var, geoName))
      return None
    jacFull = '{}{}_{}'.format(JAC_PREFIX, jacName, ch)
    jac = self.diag(jacFull)
    p, dp = self.pressure(how)
    if jac is not None and jac.shape[1] != prof.shape[1]:
      warn('{}: {} Jacobian level count {} differs from profile {}; Jacobian left blank'.format(
        self.key, var, jac.shape[1], prof.shape[1]))
      jac = None
    if jac is None:
      warn('{}: {} has no Jacobian ({} not in ydiags): profile only'.format(
        self.key, var, jacFull))
      jac = np.full(prof.shape, np.nan)
    if not (prof.shape[1] == p.shape[1]):
      warn('{}: {} level counts differ (profile {}, pressure {})'.format(
        self.key, var, prof.shape[1], p.shape[1]))
      return None
    if kind == 'T':
      return prof, jac, p, ('K', 'K/K')
    if kind == 'qv_crtm':
      return prof * 1.0e3, jac, p, ('g/kg', 'K/(g/kg)')
    if kind == 'mass_crtm':
      q = prof * GRAVITY / dp                      # kg/kg
      jq = jac * dp / GRAVITY                      # K/(kg/kg)
      return q * 1.0e3, jq / 1.0e3, p, ('g/kg', 'K/(g/kg)')
    if kind == 'kgkg':
      return prof * 1.0e3, jac / 1.0e3, p, ('g/kg', 'K/(g/kg)')
    die('internal: unknown kind ' + kind)


def alignmentCheck(ds, cfg):
  ch = int(cfg.get('channel', 8))
  if ch not in ds.channels:
    return
  bt, clr = ds.bt(ch), ds.clear(ch)
  ok = np.isfinite(bt) & np.isfinite(clr)
  if ok.sum() == 0:
    warn('{}: alignment check skipped, no valid ch{} values'.format(ds.key, ch))
    return
  frac = np.mean(np.abs(bt[ok] - clr[ok]) < 0.01)
  minFrac = float(cfg.get('min_clear_fraction', 0.01))
  msg = '{}: alignment check ch{}: {:.1%} of locations have |BT - clear BT| < 0.01 K'.format(
    ds.key, ch, frac)
  if frac < minFrac:
    msg += ' (< {:.1%}): geoval/ydiags order may not match obsout'.format(minFrac)
    if cfg.get('strict', False):
      die(msg)
    warn(msg)
  else:
    print(msg)


def expandSelections(sels):
  out = []
  for s in sels:
    if s.get('mode') == 'index' and isinstance(s.get('index'), (list, tuple)):
      for i in s['index']:
        t = dict(s)
        t['index'] = int(i)
        t['name'] = '{}_{}'.format(s['name'], i)
        out.append(t)
    else:
      out.append(s)
  return out


def inRange(x, r):
  if r is None:
    return np.ones(x.shape, dtype=bool)
  lo, hi = r
  m = np.isfinite(x)
  if lo is not None:
    m &= x >= float(lo)
  if hi is not None:
    m &= x <= float(hi)
  return m


def rangeStr(r):
  if r is None:
    return 'all'
  lo, hi = r
  if lo is None and hi is None:
    return 'all'
  if lo is None:
    return '<={:g}'.format(float(hi))
  if hi is None:
    return '>={:g}'.format(float(lo))
  return '[{:g}, {:g}]'.format(float(lo), float(hi))


def validMask(dsList, ch, qc):
  m = np.isfinite(dsList[0].obsBT(ch))
  for ds in dsList:
    m &= np.isfinite(ds.bt(ch)) & np.isfinite(ds.clear(ch))
    if qc:
      m &= ds.qcPass(ch)
  return m


def cloudProxy(clear, bt):
  return np.maximum(clear - bt, 0.0)


def select(sel, dsList, ref, ch, qc):
  '''indices of the selected locations, and a short description'''
  mode = sel.get('mode')
  m = validMask(dsList, ch, qc)
  lat, lon = ref.lat, ref.lon
  if mode == 'index':
    i = int(sel['index'])
    if i < 0 or i >= ref.nlocs:
      die('selection {}: index {} out of range [0, {})'.format(sel['name'], i, ref.nlocs))
    desc = 'Location {} ({:.2f}N, {:.2f}E)'.format(i, lat[i], lon[i])
    if not m[i]:
      warn('selection {} ch{}: location {} is missing or fails QC'.format(sel['name'], ch, i))
      return np.array([], dtype=int), desc
    return np.array([i]), desc
  if mode == 'region' or mode == 'cloud_proxy':
    if 'lat' in sel:
      m &= inRange(lat, sel['lat'])
    if 'lon' in sel:
      m &= inRange(lon, sel['lon'])
    desc = ''
    if 'lat' in sel or 'lon' in sel:
      desc = 'lat {} lon {}'.format(rangeStr(sel.get('lat')), rangeStr(sel.get('lon')))
    if mode == 'cloud_proxy':
      pch = int(sel.get('channel', 13))
      m &= validMask(dsList, pch, qc)
      mp = cloudProxy(ref.clear(pch), ref.bt(pch))
      op = cloudProxy(ref.clear(pch), ref.obsBT(pch))
      m &= inRange(mp, sel.get('model_range')) & inRange(op, sel.get('obs_range'))
      cdesc = 'ch{} proxy: model {} K, obs {} K'.format(
        pch, rangeStr(sel.get('model_range')), rangeStr(sel.get('obs_range')))
      desc = cdesc + (', ' + desc if desc else '')
    return np.where(m)[0], desc
  die('selection {}: unknown mode {}'.format(sel.get('name'), mode))


def nanmeanstd(a):
  '''mean and std over locations (axis 0), ignoring missing; all-missing levels -> nan'''
  with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=RuntimeWarning)
    return np.nanmean(a, axis=0), np.nanstd(a, axis=0)


def parseDiff(item):
  '''"A - B" or {diff: "A - B", label, color, linestyle} -> dict with keys a, b, ...'''
  spec = {'diff': item} if isinstance(item, str) else dict(item)
  parts = [s.strip() for s in str(spec.get('diff', '')).split(' - ')]
  if len(parts) != 2 or not all(parts):
    die('increment diff must look like "A - B" (spaces around -), got {!r}'.format(
      spec.get('diff')))
  spec['a'], spec['b'] = parts
  return spec


def parseJac(item, compName):
  '''"key" or {dataset: key, label, color, linestyle, linewidth} -> dict with key "dataset"'''
  spec = {'dataset': item} if isinstance(item, str) else dict(item)
  if not spec.get('dataset'):
    die('comparison {}: jacobian_datasets entry {!r} needs a dataset key'.format(compName, item))
  return spec


def comparisonKeys(comp):
  '''(jacobian specs, increment variables, diff specs, base keys), validated'''
  keys = list(comp['datasets'])
  ref = comp.get('reference', keys[0])
  jacSpecs = [parseJac(j, comp['name']) for j in comp.get('jacobian_datasets', [ref])]
  jac = [j['dataset'] for j in jacSpecs]
  inc = comp.get('increment') or {}
  incVars = list(inc.get('variables', []))
  diffs = [parseDiff(d) for d in inc.get('diffs', [])]
  base = list(inc.get('base', [ref]))
  if incVars and not diffs:
    die('comparison {}: increment.variables given without increment.diffs'.format(comp['name']))
  for d in diffs:
    # Jacobian used for the contribution decomposition (default: the B side, normally bg)
    d.setdefault('jacobian', d['b'])
  for k in jac + base + [d[s] for d in diffs for s in ('a', 'b', 'jacobian')]:
    if k not in keys:
      die('comparison {}: dataset {} (jacobian_datasets/increment) is not in its datasets {}'
          .format(comp['name'], k, keys))
  return jacSpecs, incVars, diffs, base


def setXlim(ax, lim):
  if lim is not None:
    ax.set_xlim(float(lim[0]), float(lim[1]))


def plotProfile(comp, sel, idx, desc, ch, var, dsList, ref, conf, outPath):
  pconf = conf.get('plot', {})
  how = pconf.get('vertical', 'auto')
  byKey = {ds.key: ds for ds in dsList}
  jacSpecs, incVars, diffs, baseKeys = comparisonKeys(comp)
  isInc = var in incVars

  res = {}
  for ds in dsList:
    r = ds.profileAndJacobian(var, ch, how)
    if r is not None:
      res[ds.key] = r
  if not res:
    warn('{} {} ch{} {}: nothing to plot'.format(comp['name'], sel['name'], ch, var))
    return
  units = next(iter(res.values()))[3]
  spread = pconf.get('show_spread', False) and len(idx) > 1

  def meanLine(key, which):
    '''mean/std over the selection of profile (which=0) or jacobian (which=1), with pressure'''
    a = res[key][which][idx]
    m, s = nanmeanstd(a)
    return m, s, nanmeanstd(res[key][2][idx])[0]

  def draw(ax, x, s, p, color, ls, label, lw=1.5, alpha=1.0):
    h, = ax.plot(x, p, color=color, linestyle=ls, label=label, linewidth=lw, alpha=alpha)
    if spread:
      ax.fill_betweenx(p, x - s, x + s, color=h.get_color(), alpha=0.15 * alpha, linewidth=0)
    return h

  fig, axes = plt.subplots(1, 2, sharey=True, figsize=pconf.get('figsize', [10.0, 7.5]))
  handles = []
  twin = None

  # left panel
  if not isInc:
    for ds in dsList:
      if ds.key in res:
        m, s, p = meanLine(ds.key, 0)
        handles.append(draw(axes[0], m, s, p, ds.color, ds.linestyle, ds.label))
    axes[0].set_xlabel('{} ({})'.format(var, units[0]))
    setXlim(axes[0], (pconf.get('xlim') or {}).get(var))
    axes[0].set_title('profile')
  else:
    incMax = 0.0
    for d in diffs:
      if d['a'] not in res or d['b'] not in res:
        warn('{} {}: skipping increment {} (variable missing)'.format(comp['name'], var, d['diff']))
        continue
      A, B = res[d['a']][0][idx], res[d['b']][0][idx]
      if A.shape != B.shape:
        warn('{} {}: skipping increment {} (level counts differ)'.format(
          comp['name'], var, d['diff']))
        continue
      m, s = nanmeanstd(A - B)
      p = nanmeanstd(res[d['a']][2][idx])[0]
      da = byKey[d['a']]
      label = d.get('label', '{} − {}'.format(da.label, byKey[d['b']].label))
      handles.append(draw(axes[0], m, s, p, d.get('color', da.color),
                          d.get('linestyle', '-'), label, lw=2.0))
      if np.isfinite(m).any():
        incMax = max(incMax, float(np.nanmax(np.abs(m))))
    axes[0].axvline(0.0, color='gray', linewidth=0.8)
    axes[0].set_xlabel('Δ{} ({})'.format(var, units[0]))
    lim = (pconf.get('increment_xlim') or {}).get(var)
    if lim is None and pconf.get('symmetric_increment', True) and incMax > 0:
      lim = [-1.05 * incMax, 1.05 * incMax]
    setXlim(axes[0], lim)
    # full profiles overlaid on a second (top) x axis
    baseKeys = [k for k in baseKeys if k in res]
    if baseKeys:
      twin = axes[0].twiny()
      bstyle = pconf.get('base_style') or {}
      for k in baseKeys:
        ds = byKey[k]
        m, s, p = meanLine(k, 0)
        handles.append(draw(twin, m, s, p, bstyle.get('color', ds.color),
                            bstyle.get('linestyle', ':'), ds.label + ' (top axis)',
                            lw=float(bstyle.get('linewidth', 1.2)),
                            alpha=float(bstyle.get('alpha', 0.8))))
      twin.set_xlabel('{} ({}), top axis'.format(var, units[0]), fontsize=9)
      setXlim(twin, (pconf.get('base_xlim') or {}).get(var))
    else:
      axes[0].set_title('increment')

  # right panel: Jacobian of the jacobian_datasets only (default: reference/bg);
  # a dict entry overrides the dataset's label/color/linestyle/linewidth
  noJac = []
  for j in jacSpecs:
    k = j['dataset']
    if k in res:
      ds = byKey[k]
      if not np.isfinite(res[k][1][idx]).any():   # no Jacobian for this variable: leave blank
        noJac.append(j.get('label', ds.label))
        continue
      m, s, p = meanLine(k, 1)
      draw(axes[1], m, s, p, j.get('color', ds.color), j.get('linestyle', ds.linestyle),
           j.get('label', ds.label), lw=float(j.get('linewidth', 1.5)))
  if noJac:
    axes[1].text(0.5, 0.02, 'no Jacobian: ' + ', '.join(noJac), transform=axes[1].transAxes,
                 ha='center', va='bottom', fontsize=7, color='0.4', wrap=True)
  axes[1].set_xlabel('dBT/d{} ({})'.format(var, units[1]))
  axes[1].axvline(0.0, color='gray', linewidth=0.8)
  setXlim(axes[1], (pconf.get('jacobian_xlim') or {}).get(var))
  axes[1].set_title('Jacobian')

  axes[0].set_ylabel('pressure (hPa)')
  if pconf.get('ylog', False):
    axes[0].set_yscale('log')
  pr = pconf.get('pressure_range', [1000.0, 100.0])
  axes[0].set_ylim(float(pr[0]), float(pr[1]))
  for ax in axes:
    ax.grid(True, alpha=0.3)
  # legends: below each panel (outside, default) or inside at the best location
  jacLines = [l for l in axes[1].get_lines() if not l.get_label().startswith('_')]
  lconf = pconf.get('legend') or {}
  outside = str(lconf.get('location', 'outside')).lower() == 'outside'
  lkw = dict(fontsize=float(lconf.get('fontsize', 7 if outside else 9)),
             ncol=int(lconf.get('ncol', 1)), frameon=bool(lconf.get('frame', not outside)))
  if outside:
    # anchored `offset` points below the bottom of each panel (clears the x label whatever
    # the panel height); the canvas is enlarged at save time (bbox_inches='tight') so the
    # panels keep their size
    lkw.update(loc='upper center', borderaxespad=0.0, handlelength=2.5)
    off = float(lconf.get('offset', 30.0)) / 72.0
    for ax, extra in ((axes[0], dict(handles=handles, labels=[h.get_label() for h in handles])),
                      (axes[1], {})):
      if ax is axes[1] and not jacLines:   # no Jacobian drawn: no legend
        continue
      tr = ax.transAxes + ScaledTranslation(0.0, -off, fig.dpi_scale_trans)
      ax.legend(bbox_to_anchor=(0.5, 0.0), bbox_transform=tr, **extra, **lkw)
  else:
    lkw.update(loc='best')
    axes[0].legend(handles=handles, labels=[h.get_label() for h in handles], **lkw)
    if jacLines:
      axes[1].legend(**lkw)

  # BT annotation (means over the selected locations)
  obsMean = np.nanmean(dsList[0].obsBT(ch)[idx])
  pch = int(sel.get('channel', 13)) if sel.get('mode') == 'cloud_proxy' else ch
  txt = ['Obs BT = {:.2f} K   obs proxy (vs {} clear) = {:.2f} K'.format(
    obsMean, ref.label, np.nanmean(cloudProxy(ref.clear(ch)[idx], dsList[0].obsBT(ch)[idx])))]
  for ds in dsList:
    bt, clr = ds.bt(ch)[idx], ds.clear(ch)[idx]
    txt.append('{:<12s} BT = {:.2f} K   clear-sky BT = {:.2f} K   proxy = {:.2f} K'.format(
      ds.label, np.nanmean(bt), np.nanmean(clr), np.nanmean(cloudProxy(clr, bt))))
  nTxt = len(txt)
  title = '{}  ch{}  {} ({})   {}   N = {}'.format(
    dsList[0].inst, ch, var, LONG_NAME.get(var, var), desc, len(idx))
  fig.suptitle(title, fontsize=11)
  fig.text(0.07, 0.935, '\n'.join(txt), va='top', ha='left', family='monospace', fontsize=8.5,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='0.7'))
  fig.subplots_adjust(top=0.90 - 0.025 * nTxt - (0.045 if twin is not None else 0.0),
                      bottom=0.08, left=0.08, right=0.97, wspace=0.08)
  fig.savefig(outPath, dpi=pconf.get('dpi', 150),
              **(dict(bbox_inches='tight', pad_inches=0.15) if outside else {}))
  plt.close(fig)
  print('  wrote ' + outPath)


def contribution(byKey, d, var, ch, how, idx):
  '''linearized obs-space contribution of one variable for one diff "A - B":
       per layer  c_k = J_k (x_A - x_B)_k   [K],  J from dataset d['jacobian'] (default B)
     returns (per-layer c (nsel, nlev), column sum (nsel,), pressure hPa (nsel, nlev))
     or None if the variable is unavailable in any of the three datasets'''
  r = {}
  for s in ('a', 'b', 'jacobian'):
    k = d[s]
    if k not in r:
      r[k] = byKey[k].profileAndJacobian(var, ch, how)
    if r[k] is None:
      return None
  xa, xb, jac = r[d['a']][0][idx], r[d['b']][0][idx], r[d['jacobian']][1][idx]
  if not (xa.shape == xb.shape == jac.shape):
    warn('contribution {} {}: level counts differ, skipped'.format(d['diff'], var))
    return None
  per = jac * (xa - xb)
  col = np.where(np.isfinite(per).any(axis=1), np.nansum(per, axis=1), np.nan)
  return per, col, r[d['b']][2][idx]


def contributionTable(byKey, diffs, cvars, ch, how, idx):
  '''{diff: {var: mean column contribution, 'sum': Σ, 'actual': ΔH, 'omb': O - H(B)}},
     {diff: {var: (per-layer contribution, pressure)}}'''
  table, layers = {}, {}
  for d in diffs:
    row, lay = {}, {}
    total = np.zeros(len(idx))
    anyVar = False
    for var in cvars:
      c = contribution(byKey, d, var, ch, how, idx)
      if c is None:
        row[var] = np.nan
        continue
      per, col, p = c
      row[var] = float(np.nanmean(col)) if np.isfinite(col).any() else np.nan
      lay[var] = (per, p)
      total = total + np.nan_to_num(col)
      anyVar = True
    A, B = byKey[d['a']], byKey[d['b']]
    row['sum'] = float(np.mean(total)) if anyVar else np.nan
    row['actual'] = float(np.nanmean(A.bt(ch)[idx] - B.bt(ch)[idx]))
    row['omb'] = float(np.nanmean(B.obsBT(ch)[idx] - B.bt(ch)[idx]))
    table[d['diff']] = row
    layers[d['diff']] = lay
  return table, layers


def diffLabel(d, byKey):
  return d.get('label', '{} − {}'.format(byKey[d['a']].label, byKey[d['b']].label))


def diffColor(d, byKey):
  return d.get('color', byKey[d['a']].color)


def plotContributionBars(comp, sel, perCh, diffs, cvars, byKey, conf, outPath):
  '''one panel per channel: mean column contribution of each variable, Σ, actual ΔBT, O − B'''
  cats = list(cvars) + ['Σ linear', 'ΔBT actual', 'O − B']
  keys = list(cvars) + ['sum', 'actual', 'omb']
  chs = [ch for ch in perCh if perCh[ch] is not None]
  if not chs:
    return
  ncol = min(2, len(chs))
  nrow = int(np.ceil(len(chs) / float(ncol)))
  fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 4.2 * nrow), squeeze=False)
  width = 0.8 / max(len(diffs), 1)
  x = np.arange(len(cats))
  for n, ch in enumerate(chs):
    ax = axes[n // ncol][n % ncol]
    table, nsel = perCh[ch]
    for j, d in enumerate(diffs):
      vals = np.array([table[d['diff']][k] for k in keys])
      xs = x - 0.4 + width * (j + 0.5)
      bars = ax.bar(xs, np.nan_to_num(vals), width, color=diffColor(d, byKey),
                    label=diffLabel(d, byKey), alpha=0.85)
      for b, v in zip(bars, vals):
        if np.isfinite(v):
          ax.annotate('{:.2f}'.format(v), (b.get_x() + b.get_width() / 2, v),
                      ha='center', va='bottom' if v >= 0 else 'top', fontsize=6.5,
                      xytext=(0, 1 if v >= 0 else -1), textcoords='offset points')
        else:
          ax.annotate('n/a', (b.get_x() + b.get_width() / 2, 0), ha='center', va='bottom',
                      fontsize=6.5, color='0.5')
    ax.axhline(0.0, color='gray', linewidth=0.8)
    ax.axvline(len(cvars) - 0.5, color='0.6', linewidth=0.8, linestyle=':')
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylabel('K')
    ax.set_title('ch{}   N = {}'.format(ch, nsel), fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    if n == 0:
      ax.legend(fontsize=8, loc='best')
  for n in range(len(chs), nrow * ncol):
    axes[n // ncol][n % ncol].set_visible(False)
  fig.suptitle('{}  {}: Σ_k J_k Δx_k per variable (J from {})\n{}'.format(
    comp['name'], sel['name'],
    ', '.join(sorted(set(byKey[d['jacobian']].label for d in diffs))),
    'residual (ΔBT actual − Σ linear) = nonlinearity + terms not listed '
    '(surface, O3, other hydrometeors)'), fontsize=10)
  fig.tight_layout(rect=[0, 0, 1, 1.0 - 0.6 / (4.2 * nrow)])
  fig.savefig(outPath, dpi=conf.get('plot', {}).get('dpi', 150))
  plt.close(fig)
  print('  wrote ' + outPath)


def plotContributionProfile(comp, sel, desc, ch, diffs, cvars, layers, table, nsel, byKey,
                            conf, outPath):
  '''one panel per variable: mean per-layer contribution J_k Δx_k vs pressure'''
  pconf = conf.get('plot', {})
  cconf = conf.get('contribution', {})
  vs = [v for v in cvars if any(v in layers[d['diff']] for d in diffs)]
  if not vs:
    return
  fig, axes = plt.subplots(1, len(vs), sharey=True, figsize=(3.2 * len(vs) + 1.0, 6.5),
                           squeeze=False)
  axes = axes[0]
  for ax, var in zip(axes, vs):
    for d in diffs:
      if var not in layers[d['diff']]:
        continue
      per, p = layers[d['diff']][var]
      m, _ = nanmeanstd(per)
      pm, _ = nanmeanstd(p)
      ax.plot(m, pm, color=diffColor(d, byKey), linestyle=d.get('linestyle', '-'), linewidth=1.8,
              label='{} (Σ {:.2f} K)'.format(diffLabel(d, byKey), table[d['diff']][var]))
    ax.axvline(0.0, color='gray', linewidth=0.8)
    ax.set_xlabel('J·Δ{} (K per layer)'.format(var))
    ax.set_title('{} ({})'.format(var, LONG_NAME.get(var, var)), fontsize=10)
    setXlim(ax, (cconf.get('profile_xlim') or {}).get(var))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='best')
  axes[0].set_ylabel('pressure (hPa)')
  if pconf.get('ylog', False):
    axes[0].set_yscale('log')
  pr = pconf.get('pressure_range', [1000.0, 100.0])
  axes[0].set_ylim(float(pr[0]), float(pr[1]))
  txt = []
  for d in diffs:
    r = table[d['diff']]
    txt.append('{}: Σ linear {:.2f} K, ΔBT actual {:.2f} K, O − B {:.2f} K'.format(
      diffLabel(d, byKey), r['sum'], r['actual'], r['omb']))
  fig.suptitle('{}  ch{}  per-layer contribution   {}   N = {}\n{}'.format(
    byKey[diffs[0]['a']].inst, ch, desc, nsel, '\n'.join(txt)), fontsize=9)
  fig.tight_layout(rect=[0, 0, 1, 0.97 - 0.03 * len(txt)])
  fig.savefig(outPath, dpi=pconf.get('dpi', 150))
  plt.close(fig)
  print('  wrote ' + outPath)


def printContribution(compName, selName, ch, table, cvars, nsel, byKey, diffs):
  cols = list(cvars) + ['Σlin', 'ΔBT', 'O−B']
  keys = list(cvars) + ['sum', 'actual', 'omb']
  print('  contribution {} {} ch{} (N = {}), K:'.format(compName, selName, ch, nsel))
  print('    {:<26s}'.format('') + ''.join('{:>9s}'.format(c) for c in cols))
  for d in diffs:
    r = table[d['diff']]
    print('    {:<26s}'.format(diffLabel(d, byKey)[:26]) +
          ''.join('{:>9.3f}'.format(r[k]) if np.isfinite(r[k]) else '{:>9s}'.format('n/a')
                  for k in keys))


def useCartopy(mconf):
  want = mconf.get('use_cartopy', 'auto')
  if want is False or str(want).lower() == 'false':
    return None
  try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    return ccrs, cfeature
  except Exception as e:
    if want is True or str(want).lower() == 'true':
      die('cartopy requested but not importable: {}'.format(e))
    return None


def plotMap(comp, sel, dsList, ref, conf, outPath):
  mconf = conf.get('map', {})
  qc = conf.get('qc', True)
  ch = int(mconf.get('channel', 13))
  idx, desc = select(sel, dsList, ref, ch, qc)
  if sel.get('mode') == 'index':
    # mark the chosen location even if it fails QC / is missing at the map channel
    idx = np.array([int(sel['index'])])
  show = np.isfinite(ref.lat) & np.isfinite(ref.lon)
  if mconf.get('qc_only', False):
    show &= validMask(dsList, ch, True)
  model, obs = ref.bt(ch), ref.obsBT(ch)
  vals = np.concatenate([model[show & np.isfinite(model)], obs[show & np.isfinite(obs)]])
  vmin = mconf.get('vmin', None)
  vmax = mconf.get('vmax', None)
  if vals.size:
    vmin = np.percentile(vals, 1) if vmin is None else vmin
    vmax = np.percentile(vals, 99) if vmax is None else vmax

  cp = useCartopy(mconf)
  sub = {}
  if cp is not None:
    sub = dict(projection=cp[0].PlateCarree())
  fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), subplot_kw=sub)
  ext = mconf.get('extent', None)
  if ext is None and show.any():
    ext = [np.nanmin(ref.lon[show]), np.nanmax(ref.lon[show]),
           np.nanmin(ref.lat[show]), np.nanmax(ref.lat[show])]
  ms = float(mconf.get('marker_size', 6))
  tkw = {}
  if cp is not None:
    tkw = dict(transform=cp[0].PlateCarree())
  for ax, data, name in [(axes[0], model, '{} model BT'.format(ref.label)),
                         (axes[1], obs, 'Obs BT')]:
    s = show & np.isfinite(data)
    sc = ax.scatter(ref.lon[s], ref.lat[s], c=data[s], s=ms, cmap=mconf.get('cmap', 'jet_r'),
                    vmin=vmin, vmax=vmax, linewidths=0, **tkw)
    if len(idx):
      if sel.get('mode') == 'index':
        ax.scatter(ref.lon[idx], ref.lat[idx], marker='*', s=220, facecolor='magenta',
                   edgecolor='k', linewidths=0.8, zorder=5, **tkw)
      else:
        ax.scatter(ref.lon[idx], ref.lat[idx], s=ms * 3, facecolor='none', edgecolor='k',
                   linewidths=0.6, zorder=5, **tkw)
    if 'lat' in sel and 'lon' in sel and sel.get('mode') != 'index':
      la, lo = sel['lat'], sel['lon']
      la = [ext[2] if la[0] is None else la[0], ext[3] if la[1] is None else la[1]]
      lo = [ext[0] if lo[0] is None else lo[0], ext[1] if lo[1] is None else lo[1]]
      ax.add_patch(Rectangle((lo[0], la[0]), lo[1] - lo[0], la[1] - la[0], fill=False,
                             edgecolor='magenta', linewidth=1.5, zorder=6, **tkw))
    if cp is not None:
      ax.coastlines(linewidth=0.6)
      if ext is not None:
        ax.set_extent(ext, crs=cp[0].PlateCarree())
      gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
      gl.top_labels = False
      gl.right_labels = False
    else:
      if ext is not None:
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
      ax.set_xlabel('longitude')
      ax.set_ylabel('latitude')
      ax.set_aspect('equal', adjustable='box')
      ax.grid(True, alpha=0.3)
    ax.set_title('{} ch{}'.format(name, ch))
  fig.suptitle('{}  {}   {}   N selected = {}'.format(
    ref.inst, sel['name'], desc, len(idx)), fontsize=11)
  fig.subplots_adjust(bottom=0.2, top=0.88, left=0.06, right=0.97, wspace=0.15)
  cax = fig.add_axes([0.25, 0.08, 0.5, 0.03])
  cb = fig.colorbar(sc, cax=cax, orientation='horizontal', extend='both')
  cb.set_label('brightness temperature (K)')
  fig.savefig(outPath, dpi=conf.get('plot', {}).get('dpi', 150))
  plt.close(fig)
  print('  wrote ' + outPath)


def jacobianPeakSummary(ds, channels, how):
  '''pressure of the peak |dBT/dT| of the mean profile, as a quick order/unit sanity check'''
  out = []
  for ch in channels:
    r = ds.profileAndJacobian('T', ch, how)
    if r is None:
      continue
    _, jac, p, _ = r
    jm, _ = nanmeanstd(np.abs(jac))
    pm, _ = nanmeanstd(p)
    if np.all(np.isnan(jm)):
      continue
    out.append('ch{} {:.0f} hPa'.format(ch, pm[np.nanargmax(jm)]))
  if out:
    print('{}: mean |dBT/dT| peaks at '.format(ds.key) + ', '.join(out))


def diagnose(ds, variables, channels, qcPass):
  '''raw (unconverted) statistics, to tell "Jacobian really 0" from "small on the plot axis"'''
  print('\n=== diagnose {} ({}) ==='.format(ds.key, ds.operator))
  cf = ds.geo('cloud_area_fraction_in_atmosphere_layer')
  for var in variables:
    if var == 'T':
      continue
    geoName, jacName, _ = VARMAP[ds.operator][var]
    prof = ds.geo(geoName)
    if prof is None:
      print('{:3s} geoval {}: NOT in geoval files'.format(var, geoName))
      continue
    cloudy = prof > (1.0e-10 if ds.operator == 'rttov' or var == 'qv' else 1.0e-9)
    line = '{:3s} geoval {}: max {:.3e}, layers > tiny {:.2%}'.format(
      var, geoName, np.nanmax(prof), np.mean(cloudy))
    if cf is not None and cf.shape == prof.shape and var != 'qv':
      line += ', of which cloud_area_fraction == 0: {:.2%}'.format(
        np.mean(cf[cloudy] <= 0.0) if cloudy.any() else 0.0)
    print(line)
    for ch in channels:
      jac = ds.diag('{}{}_{}'.format(JAC_PREFIX, jacName, ch))
      if jac is None:
        print('    ch{}: Jacobian {} NOT in ydiags'.format(ch, jacName))
        continue
      nz = np.isfinite(jac) & (jac != 0.0)
      m = qcPass(ch)
      print('    ch{}: raw Jacobian min {:.3e} max {:.3e}, nonzero {:.2%} of all values, '
            '{:.2%} of profiles have any nonzero (QC-passed: {:.2%}), nonzero where cloudy {:.2%}'
            .format(ch, np.nanmin(jac), np.nanmax(jac), np.mean(nz), np.mean(nz.any(axis=1)),
                    np.mean(nz[m].any(axis=1)) if m.any() else 0.0,
                    np.mean(nz[cloudy]) if cloudy.any() else 0.0))
  if cf is None:
    print('cloud_area_fraction_in_atmosphere_layer: NOT in geoval files')
  else:
    print('cloud_area_fraction_in_atmosphere_layer: max {:.3f}, layers > 0 {:.2%}, '
          'profiles with any > 0 {:.2%}'.format(np.nanmax(cf), np.mean(cf > 0.0),
                                                 np.mean((cf > 0.0).any(axis=1))))
  for ch in channels:
    d = ds.bt(ch) - ds.clear(ch)
    ok = np.isfinite(d)
    print('ch{}: BT - clear-sky BT: min {:.2f} K, mean {:.2f} K, |diff| > 0.1 K at {:.2%} '
          'of locations'.format(ch, np.nanmin(d), np.nanmean(d), np.mean(np.abs(d[ok]) > 0.1)))


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('config', help='yaml configuration')
  ap.add_argument('--diagnose', action='store_true',
                  help='print raw hydrometeor / cloud fraction / Jacobian statistics per dataset '
                       'and exit (no figures)')
  args = ap.parse_args()
  with open(args.config) as f:
    conf = yaml.safe_load(f)

  templates = {k: str(v) for k, v in (conf.get('templates') or {}).items()}
  inst = conf['instrument']
  app = conf.get('app', 'hofx')
  channels = [int(c) for c in conf['channels']]
  variables = list(conf['variables'])
  qc = conf.get('qc', True)
  how = conf.get('plot', {}).get('vertical', 'auto')
  selections = expandSelections(conf['selections'])
  comparisons = conf['comparisons']
  outDir = fmt(conf.get('output', {}).get('dir', './figures'), templates)
  ext = conf.get('output', {}).get('format', 'png')

  # read only the datasets used by a comparison
  used = []
  for c in comparisons:
    for k in c['datasets']:
      if k not in conf['datasets']:
        die('comparison {}: unknown dataset {}'.format(c['name'], k))
      if k not in used:
        used.append(k)
  data = {k: Dataset(k, conf['datasets'][k], inst, app, templates) for k in used}
  if args.diagnose:
    for ds in data.values():
      diagnose(ds, variables, channels, ds.qcPass)
    return
  for ds in data.values():
    alignmentCheck(ds, conf.get('alignment_check', {}))
    jacobianPeakSummary(ds, channels, how)

  for c in comparisons:
    dsList = [data[k] for k in c['datasets']]
    ref = data[c.get('reference', c['datasets'][0])]
    for ds in dsList[1:]:
      if ds.nlocs != dsList[0].nlocs or not (
          np.allclose(ds.lat, dsList[0].lat, atol=1e-4, equal_nan=True) and
          np.allclose(ds.lon, dsList[0].lon, atol=1e-4, equal_nan=True)):
        die('comparison {}: {} and {} have different obs locations'.format(
          c['name'], dsList[0].key, ds.key))
    # validate jacobian_datasets / increment before plotting
    _, _, diffs, _ = comparisonKeys(c)
    byKey = {ds.key: ds for ds in dsList}
    cconf = conf.get('contribution') or {}
    doContrib = bool(cconf.get('enabled', False)) and len(diffs) > 0
    cvars = list(cconf.get('variables') or variables)
    print('comparison {}: {} (reference {})'.format(c['name'], c['datasets'], ref.key))
    for sel in selections:
      d = os.path.join(outDir, c['name'], sel['name'])
      os.makedirs(d, exist_ok=True)
      perCh = {}
      for ch in channels:
        idx, desc = select(sel, dsList, ref, ch, qc)
        print(' {} ch{}: {} locations'.format(sel['name'], ch, len(idx)))
        if len(idx) == 0:
          continue
        for var in variables:
          plotProfile(c, sel, idx, desc, ch, var, dsList, ref, conf,
                      os.path.join(d, '{}_ch{}_{}.{}'.format(inst, ch, var, ext)))
        if doContrib:
          table, layers = contributionTable(byKey, diffs, cvars, ch, how, idx)
          perCh[ch] = (table, len(idx))
          printContribution(c['name'], sel['name'], ch, table, cvars, len(idx), byKey, diffs)
          if cconf.get('profile', True):
            plotContributionProfile(c, sel, desc, ch, diffs, cvars, layers, table, len(idx),
                                    byKey, conf, os.path.join(
                                      d, '{}_ch{}_contribution_profile.{}'.format(inst, ch, ext)))
      if doContrib and perCh:
        plotContributionBars(c, sel, perCh, diffs, cvars, byKey, conf,
                             os.path.join(d, '{}_contribution.{}'.format(inst, ext)))
      if conf.get('map', {}).get('enabled', True):
        plotMap(c, sel, dsList, ref, conf, os.path.join(
          d, '{}_map_ch{}.{}'.format(inst, int(conf.get('map', {}).get('channel', 13)), ext)))


if __name__ == '__main__':
  main()
