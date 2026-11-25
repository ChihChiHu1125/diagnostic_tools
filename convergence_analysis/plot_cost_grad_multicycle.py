#!/home/users/industry/nea/hucc/miniconda3/bin/python
"""
Multi-cycle convergence diagnostics for MPAS-JEDI variational runs.

Reads <base>/<yyyymmddhh>/run/mem001/jedi.log for every cycle and produces:

  A  <prefix>_A_normred.png   Norm reduction vs inner iteration, one panel per
                              outer loop; all cycles overlaid + median / IQR.
  B  <prefix>_B_cost.png      Quadratic J, Jo, Jb normalised by J0 (nonlinear J
                              at the first guess of outer loop 0) vs cumulative
                              iteration; nonlinear Jo marked at outer-loop
                              boundaries to show the linearisation gap.
  C  <prefix>_C_summary.png   Per-cycle time series: final norm reduction,
                              nonlinear Jo/n, nonlinearity ratio, nobs.
  D  <prefix>_D_panels.png    (optional, --panels) small multiples, one per cycle.
  <prefix>_summary.csv        Per-cycle numbers behind figure C.

Notes
  * Only lines inside the minimizer section (e.g. "OOPS_STATS DRPCG start" ...
    "OOPS_STATS DRPCG end") are used, so GMRESR "Norm reduction" lines (from the
    B^-1 solve used for the nonlinear Jb) are excluded.
  * The printed "Nonlinear Jb" comes from an unconverged GMRESR B^-1 solve and is
    not reliable, so Jb is always taken from the quadratic cost function.

Usage
  ./plot_cost_grad_multicycle.py convergence.yaml                 # settings from YAML
  ./plot_cost_grad_multicycle.py convergence.yaml --end 2024091620  # CLI overrides YAML
  ./plot_cost_grad_multicycle.py --start 2024091613 --end 2024091712 \
      --base /scratch/.../CyclingDA --prefix conv_offlineBC --panels

YAML keys (all optional; precedence: built-in defaults < YAML < command line)
  base:     /scratch/.../CyclingDA     # experiment CyclingDA directory
  start:    2024091613                 # first cycle
  end:      2024091712                 # last cycle
  interval: 1                          # hours between cycles
  subpath:  run/mem001/jedi.log        # log path under <base>/<cycle>
  prefix:   convergence                # output file prefix
  outdir:   .                          # output directory (created if missing; ~ and $VAR expanded)
  panels:   false                      # also make figure D
"""

import argparse
import csv
import datetime as dt
import os
import re
import sys

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_BASE = ('/scratch/users/industry/nea/hucc/regional_mpasjedi/'
                'hucc_3dvar_regional_vbal_crtm_AllSky_AHI_offlineBC/CyclingDA')
DEFAULT_SUBPATH = 'run/mem001/jedi.log'

RE_ALGO = re.compile(r'Minimizer algorithm\s*=\s*(\S+)')
RE_SEC = re.compile(r'OOPS_STATS\s+(\S+)\s+(start|end)\s')
RE_MAXIT = re.compile(r'Minimizer: max iter = (\d+), requested norm reduction = (\S+)')
RE_NORMRED = re.compile(r'Norm reduction \(\s*(\d+)\)\s*=\s*(\S+)')
RE_QJ = re.compile(r'Quadratic cost function: (J|Jb|JoJc)\s*\(\s*(\d+)\)\s*=\s*(\S+)')
RE_JO_OBS = re.compile(r'Nonlinear Jo\((.+?)\)\s*=\s*(\S+), nobs = (\d+)')
RE_JO_TOT = re.compile(r'Nonlinear Jo\s*=\s*(\S+)')
RE_J_TOT = re.compile(r'CostFunction: Nonlinear J\s*=\s*(\S+)')


# ----------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------
class Outer(object):
    def __init__(self):
        self.it = []        # inner iteration index
        self.normred = []
        self.J = []
        self.Jb = []
        self.Jo = []        # JoJc

    def finalize(self):
        for k in ('it', 'normred', 'J', 'Jb', 'Jo'):
            setattr(self, k, np.asarray(getattr(self, k), dtype=float))
        return self


def parse_log(path):
    """Return dict with per-outer-loop arrays and nonlinear Jo evaluations."""
    algo = None
    maxit, target = None, None
    outers = []
    cur = None
    in_min = False
    jo_nl, nobs_nl = [], []    # one entry per nonlinear Jo evaluation
    nobs_acc = 0
    j_nl = []

    with open(path, errors='replace') as f:
        for line in f:
            if algo is None:
                m = RE_ALGO.search(line)
                if m:
                    algo = m.group(1)
                    continue
            m = RE_SEC.search(line)
            if m and algo is not None and m.group(1) == algo:
                if m.group(2) == 'start':
                    in_min = True
                    cur = Outer()
                else:
                    in_min = False
                    if cur is not None:
                        outers.append(cur.finalize())
                    cur = None
                continue
            if in_min:
                m = RE_NORMRED.search(line)
                if m:
                    cur.it.append(int(m.group(1)))
                    cur.normred.append(float(m.group(2)))
                    continue
                m = RE_QJ.search(line)
                if m:
                    {'J': cur.J, 'Jb': cur.Jb, 'JoJc': cur.Jo}[m.group(1)].append(float(m.group(3)))
                    continue
            else:
                m = RE_MAXIT.search(line)
                if m:
                    maxit, target = int(m.group(1)), float(m.group(2))
                    continue
                m = RE_JO_OBS.search(line)
                if m:
                    nobs_acc += int(m.group(3))
                    continue
                m = RE_JO_TOT.search(line)
                if m:
                    jo_nl.append(float(m.group(1)))
                    nobs_nl.append(nobs_acc)
                    nobs_acc = 0
                    continue
                m = RE_J_TOT.search(line)
                if m:
                    j_nl.append(float(m.group(1)))

    # an unterminated minimizer (crashed run) still counts
    if cur is not None and len(cur.it) > 0:
        outers.append(cur.finalize())

    # keep only iterations with complete cost info
    for o in outers:
        n = min(len(o.it), len(o.J), len(o.Jb), len(o.Jo))
        o.it, o.normred, o.J, o.Jb, o.Jo = o.it[:n], o.normred[:n], o.J[:n], o.Jb[:n], o.Jo[:n]

    return dict(algo=algo, maxit=maxit, target=target, outers=outers,
                jo_nl=np.asarray(jo_nl), nobs=np.asarray(nobs_nl), j_nl=np.asarray(j_nl))


def cycle_list(start, end, hours):
    t0 = dt.datetime.strptime(start, '%Y%m%d%H')
    t1 = dt.datetime.strptime(end, '%Y%m%d%H')
    out = []
    while t0 <= t1:
        out.append(t0.strftime('%Y%m%d%H'))
        t0 += dt.timedelta(hours=hours)
    return out


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def pad_stack(arrs):
    n = max([len(a) for a in arrs]) if arrs else 0
    out = np.full((len(arrs), n), np.nan)
    for i, a in enumerate(arrs):
        out[i, :len(a)] = a
    return out


def cycle_colors(n, cmap='viridis'):
    cm = plt.get_cmap(cmap)
    return [cm(x) for x in np.linspace(0.0, 0.95, max(n, 1))]


def add_cycle_colorbar(fig, axes, cycles, cmap='viridis'):
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap),
                               norm=matplotlib.colors.Normalize(0, 0.95 * max(len(cycles), 1)))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, pad=0.015, fraction=0.025)
    n = len(cycles)
    step = max(1, int(np.ceil(n / 8.0)))
    idx = list(range(0, n, step))
    if (n - 1) not in idx:
        idx.append(n - 1)
    cb.set_ticks([i * 0.95 for i in idx] if n > 1 else [0])
    cb.set_ticklabels([cycles[i] for i in idx])
    cb.set_label('cycle')
    return cb


def iqr_band(ax, x, stack, color='k', label='median'):
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            med = np.nanmedian(stack, axis=0)
            q1 = np.nanpercentile(stack, 25, axis=0)
            q3 = np.nanpercentile(stack, 75, axis=0)
    ax.fill_between(x, q1, q3, color=color, alpha=0.15, lw=0,
                    label='IQR' if label is not None else None)
    ax.plot(x, med, color=color, lw=2.2, label=label)


def offsets(data):
    """Cumulative x offset of each outer loop (shared across cycles)."""
    nout = max([len(d['outers']) for d in data.values()])
    lens = np.zeros(nout, dtype=int)
    for d in data.values():
        for k, o in enumerate(d['outers']):
            lens[k] = max(lens[k], len(o.it))
    return np.concatenate([[0], np.cumsum(lens)])[:-1], lens


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def fig_A(data, cycles, out, target):
    nout = max([len(d['outers']) for d in data.values()])
    cols = cycle_colors(len(cycles))
    fig, axes = plt.subplots(1, nout, figsize=(5.2 * nout + 1.0, 4.4), sharey=True, squeeze=False)
    axes = axes[0]
    for k in range(nout):
        ax = axes[k]
        stack = []
        for i, c in enumerate(cycles):
            if c not in data or k >= len(data[c]['outers']):
                continue
            o = data[c]['outers'][k]
            ax.plot(o.it, o.normred, color=cols[i], lw=0.8, alpha=0.75)
            stack.append(o.normred)
        if stack:
            st = pad_stack(stack)
            iqr_band(ax, np.arange(st.shape[1]), st)
        if target:
            ax.axhline(target, color='r', ls='--', lw=1, label='target %g' % target)
        ax.axhline(1.0, color='0.5', ls=':', lw=0.8)
        ax.set_yscale('log')
        ax.set_title('Outer loop %d' % (k + 1))
        ax.set_xlabel('Inner iteration')
        ax.grid(True, which='both', alpha=0.3)
    axes[0].set_ylabel('Norm reduction  |r_k| / |r_0|')
    axes[0].legend(loc='lower left', fontsize=8)
    add_cycle_colorbar(fig, list(axes), cycles)
    fig.suptitle('Minimizer convergence, %d cycles' % len(data), y=1.0)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)


def fig_B(data, cycles, out):
    offs, lens = offsets(data)
    nout = len(offs)
    cols = cycle_colors(len(cycles))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), squeeze=False)
    axes = axes[0]
    names = [('J', 'Quadratic J / J0'), ('Jo', 'Quadratic Jo / J0  (o = nonlinear Jo)'),
             ('Jb', 'Quadratic Jb / J0')]
    xmax = offs[-1] + lens[-1] + 1
    for p, (key, title) in enumerate(names):
        ax = axes[p]
        for k in range(nout):
            stack = []
            for i, c in enumerate(cycles):
                if c not in data or k >= len(data[c]['outers']):
                    continue
                d = data[c]
                J0 = d['j_nl'][0] if len(d['j_nl']) else d['outers'][0].J[0]
                o = d['outers'][k]
                y = getattr(o, key) / J0
                ax.plot(offs[k] + o.it, y, color=cols[i], lw=0.8, alpha=0.75)
                stack.append(y)
            if stack:
                st = pad_stack(stack)
                iqr_band(ax, offs[k] + np.arange(st.shape[1]), st,
                         label='median' if k == 0 else None)
            if k > 0:
                ax.axvline(offs[k] - 0.5, color='0.4', ls=':', lw=1)
        if key == 'Jo':
            # nonlinear Jo: start of each outer loop + final analysis
            xs = list(offs) + [xmax - 0.5]
            for i, c in enumerate(cycles):
                if c not in data:
                    continue
                d = data[c]
                J0 = d['j_nl'][0] if len(d['j_nl']) else d['outers'][0].J[0]
                jo = d['jo_nl'] / J0
                n = min(len(jo), len(xs))
                ax.plot(xs[:n], jo[:n], 'o', ms=4, mfc='none', mec=cols[i], mew=1.0)
            ax.plot([], [], 'o', mfc='none', mec='k', label='nonlinear Jo')
        ax.set_title(title)
        ax.set_xlabel('Cumulative inner iteration')
        ax.set_xlim(-1, xmax + 1)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
    axes[1].legend(loc='upper right', fontsize=8)
    add_cycle_colorbar(fig, list(axes), cycles)
    fig.suptitle('Cost function normalised by J0 (nonlinear J at first guess)', y=1.0)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)


def summarize(data, cycles):
    nout = max([len(d['outers']) for d in data.values()])
    rows = []
    for c in cycles:
        r = dict(cycle=c)
        d = data.get(c)
        for k in range(nout):
            r['final_normred_o%d' % (k + 1)] = np.nan
            r['niter_o%d' % (k + 1)] = np.nan
            r['quadJo_end_o%d' % (k + 1)] = np.nan
            r['quadJb_end_o%d' % (k + 1)] = np.nan
            r['nonlin_ratio_o%d' % (k + 1)] = np.nan
        for k in range(nout + 1):
            r['nlJo_%d' % k] = np.nan
            r['nlJo_per_n_%d' % k] = np.nan
        r['nobs'] = np.nan
        if d is not None:
            for k, o in enumerate(d['outers']):
                if len(o.it) == 0:
                    continue
                r['final_normred_o%d' % (k + 1)] = o.normred[-1]
                r['niter_o%d' % (k + 1)] = o.it[-1]
                r['quadJo_end_o%d' % (k + 1)] = o.Jo[-1]
                r['quadJb_end_o%d' % (k + 1)] = o.Jb[-1]
                if k + 1 < len(d['jo_nl']):
                    r['nonlin_ratio_o%d' % (k + 1)] = d['jo_nl'][k + 1] / o.Jo[-1]
            for k in range(min(nout + 1, len(d['jo_nl']))):
                r['nlJo_%d' % k] = d['jo_nl'][k]
                if d['nobs'][k] > 0:
                    r['nlJo_per_n_%d' % k] = d['jo_nl'][k] / d['nobs'][k]
            if len(d['nobs']):
                r['nobs'] = d['nobs'][0]
        rows.append(r)
    return rows, nout


def write_csv(rows, out):
    keys = list(rows[0].keys())
    with open(out, 'w') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(dict((k, ('%.6g' % v) if isinstance(v, float) else v) for k, v in r.items()))


def fig_C(rows, nout, out, target):
    x = np.arange(len(rows))
    labels = [r['cycle'] for r in rows]
    g = lambda key: np.array([r[key] for r in rows], dtype=float)
    ocol = plt.get_cmap('tab10')

    fig, axes = plt.subplots(4, 1, figsize=(max(8, 0.45 * len(rows) + 3), 11), sharex=True)

    ax = axes[0]
    for k in range(nout):
        ax.plot(x, g('final_normred_o%d' % (k + 1)), 'o-', color=ocol(k), label='outer %d' % (k + 1))
    if target:
        ax.axhline(target, color='r', ls='--', lw=1, label='target %g' % target)
    ax.set_yscale('log')
    ax.set_ylabel('Final norm\nreduction')
    ax.legend(fontsize=8, ncol=nout + 1)

    ax = axes[1]
    for k in range(nout + 1):
        lab = 'background' if k == 0 else ('analysis' if k == nout else 'start outer %d' % (k + 1))
        ax.plot(x, g('nlJo_per_n_%d' % k), 'o-', color=plt.get_cmap('plasma')(k / max(nout, 1) * 0.85),
                label=lab)
    ax.set_ylabel('Nonlinear Jo / n')
    ax.legend(fontsize=8, ncol=nout + 1)

    ax = axes[2]
    for k in range(nout):
        ax.plot(x, g('nonlin_ratio_o%d' % (k + 1)), 'o-', color=ocol(k), label='outer %d' % (k + 1))
    ax.axhline(1.0, color='0.4', ls=':', lw=1)
    ax.set_ylabel('Nonlinear Jo /\nquadratic Jo(end)')
    ax.legend(fontsize=8)

    ax = axes[3]
    ax.bar(x, g('nobs'), color='0.6')
    ax.set_ylabel('nobs (assimilated)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=8)
    ax.set_xlabel('Cycle')

    for ax in axes:
        ax.grid(True, which='both', alpha=0.3)
    fig.suptitle('Per-cycle convergence summary')
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)


def fig_D(data, cycles, out):
    avail = [c for c in cycles if c in data]
    n = len(avail)
    ncol = min(6, n)
    nrow = int(np.ceil(n / float(ncol)))
    offs, lens = offsets(data)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow), squeeze=False)
    for i, c in enumerate(avail):
        ax = axes[i // ncol][i % ncol]
        ax2 = ax.twinx()
        d = data[c]
        for k, o in enumerate(d['outers']):
            xx = offs[k] + o.it
            ax.plot(xx, o.J, 'r-', lw=1, label='J' if k == 0 else None)
            ax.plot(xx, o.Jb, 'b-', lw=1, label='Jb' if k == 0 else None)
            ax.plot(xx, o.Jo, 'k-', lw=1, label='Jo' if k == 0 else None)
            ax2.semilogy(xx, o.normred, 'g:', lw=1)
        ax.set_title(c, fontsize=9)
        ax.tick_params(labelsize=7)
        ax2.tick_params(labelsize=7, colors='g')
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 3))
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')
    fig.suptitle('Per-cycle cost (left: J red, Jb blue, Jo black) and norm reduction (right, green)')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------------
DEFAULTS = dict(base=DEFAULT_BASE, subpath=DEFAULT_SUBPATH, start='2024091613', end='2024091712',
                interval=1, prefix='convergence', outdir='.', panels=False)


def load_config(path):
    try:
        import yaml
    except ImportError:
        sys.exit('PyYAML is required for --config (pip/conda install pyyaml)')
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        sys.exit('Unknown key(s) in %s: %s (allowed: %s)'
                 % (path, ', '.join(sorted(unknown)), ', '.join(sorted(DEFAULTS))))
    # YAML may read 2024091613 as an int
    for k in ('start', 'end'):
        if k in cfg:
            cfg[k] = str(cfg[k])
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('config', nargs='?', help='YAML config file (keys: %s)' % ', '.join(DEFAULTS))
    p.add_argument('--base', help='CyclingDA directory')
    p.add_argument('--subpath', help='log path relative to <base>/<cycle>')
    p.add_argument('--start', help='first cycle yyyymmddhh')
    p.add_argument('--end', help='last cycle yyyymmddhh')
    p.add_argument('--interval', type=int, help='cycle interval [h]')
    p.add_argument('--prefix', help='output file prefix')
    p.add_argument('--outdir', help='output directory (created if missing)')
    p.add_argument('--panels', action='store_true', default=None,
                   help='also make per-cycle small multiples (D)')
    args = p.parse_args()

    # precedence: built-in defaults < YAML < command line
    opts = dict(DEFAULTS)
    if args.config:
        opts.update(load_config(args.config))
    for k in DEFAULTS:
        v = getattr(args, k)
        if v is not None:
            opts[k] = v
    opts['interval'] = int(opts['interval'])
    opts['panels'] = bool(opts['panels'])
    opts['outdir'] = os.path.expandvars(os.path.expanduser(str(opts['outdir'])))
    a = argparse.Namespace(**opts)
    print('base=%s  cycles=%s..%s every %dh  outdir=%s  prefix=%s'
          % (a.base, a.start, a.end, a.interval, a.outdir, a.prefix))
    if not os.path.isdir(a.outdir):
        os.makedirs(a.outdir)
    out = os.path.join(a.outdir, a.prefix)

    cycles = cycle_list(a.start, a.end, a.interval)
    data = {}
    for c in cycles:
        path = os.path.join(a.base, c, a.subpath)
        if not os.path.isfile(path):
            print('WARNING: missing %s' % path)
            continue
        d = parse_log(path)
        if not d['outers']:
            print('WARNING: no minimizer iterations found in %s' % path)
            continue
        data[c] = d
        print('%s  %s  outer=%d  iters=%s  final normred=%s  nonlinear Jo=%s' % (
            c, d['algo'], len(d['outers']),
            [int(o.it[-1]) for o in d['outers']],
            ['%.3g' % o.normred[-1] for o in d['outers']],
            ['%.1f' % v for v in d['jo_nl']]))
    if not data:
        sys.exit('No usable logs found.')

    target = next((d['target'] for d in data.values() if d['target']), None)

    fig_A(data, cycles, out + '_A_normred.png', target)
    fig_B(data, cycles, out + '_B_cost.png')
    rows, nout = summarize(data, cycles)
    write_csv(rows, out + '_summary.csv')
    fig_C(rows, nout, out + '_C_summary.png', target)
    if a.panels:
        fig_D(data, cycles, out + '_D_panels.png')
    print('Done: %s_{A_normred,B_cost,C_summary%s}.png, %s_summary.csv'
          % (out, ',D_panels' if a.panels else '', out))


if __name__ == '__main__':
    main()
