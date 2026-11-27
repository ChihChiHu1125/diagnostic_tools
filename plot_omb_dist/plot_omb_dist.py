#!/usr/bin/env python
"""
Multi-cycle OmB / normalized-OmB distribution for Himawari AHI channels.

  OmB            = ObsValue - model BT   (all variables: <group>/brightnessTemperature)
  model BT       = bias_correction: true  -> bias-corrected H(x)
                   bias_correction: false -> H(x) without bias correction
                   (bias.hofx_includes_bias: true  -> hofx is H(x) + ObsBias, the JEDI
                    default, so raw H(x) = hofx - ObsBias)
  normalized OmB = OmB / EffectiveError/brightnessTemperature
  selection      : EffectiveQC/brightnessTemperature == qc_value (default 0)
  CA             = (|B - Bclr| + |O - Bclr|) / 2, symmetric cloud impact (Okamoto et al.
                   2014), B = H(x) and Bclr = ClearSkyHofX, both WITHOUT bias correction,
                   or Bclr = fixed value per channel (ca.method: fixed)

Figures (<tag> = BC|noBC_<first cycle>-<last cycle>_<lead>hr):
  <prefix>_OmB_<tag>        OmB histogram, one panel per channel
  <prefix>_NormOmB_<tag>    normalized OmB histogram, one panel per channel
  <prefix>_scatter_<tag>    (plot.scatter.enabled) model BT (x) vs observed BT (y),
                            coloured by point density (log scale);
                            rows = experiments, columns = plot.scatter.channels
  <prefix>_boxplot_<tag>    (plot.boxplot.enabled) Okamoto (2017, QJRMS, Fig. 11)-style
                            boxplots of observed / model BT and of O-B; one row per
                            channel; optionally before QC (plot.boxplot.apply_qc: false)
  <prefix>_CA-OmB_<tag>     (plot.cloud_impact.enabled) Okamoto (2017, Figs 3-4)-style
                            density plot of CA (x) and O-B (y), coloured by log10(count);
                            rows = experiments, columns = plot.cloud_impact.channels

Usage:
  python plot_omb_dist.py plot_omb_dist.yaml
"""
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import yaml
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

GROUP_VAR = "brightnessTemperature"
FILL_THRESH = 1.0e30   # float fill is -3.368795e+38


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def build_cycles(spec):
    if isinstance(spec, (list, tuple)):
        return [str(c) for c in spec]
    t0 = datetime.strptime(str(spec["start"]), "%Y%m%d%H")
    t1 = datetime.strptime(str(spec["end"]), "%Y%m%d%H")
    dt = timedelta(hours=int(spec.get("interval_hours", 6)))
    out = []
    t = t0
    while t <= t1:
        out.append(t.strftime("%Y%m%d%H"))
        t += dt
    return out


def normalize_experiments(cfg):
    exps = cfg.get("experiments")
    if exps is None:
        exps = [cfg["experiment"]]
    out = []
    for e in exps:
        if isinstance(e, str):
            out.append({"name": e, "label": e})
        else:
            out.append({"name": e["name"], "label": e.get("label", e["name"])})
    return out


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def _ok(a):
    return np.isfinite(a) & (np.abs(a) < FILL_THRESH)


def bias_settings(cfg):
    """(bias_correction, bias group, hofx_includes_bias) from the yaml."""
    bcfg = cfg.get("bias", {}) or {}
    return (bool(cfg.get("bias_correction", True)),
            bcfg.get("group", "ObsBias"),
            bool(bcfg.get("hofx_includes_bias", True)))


def bias_text(cfg):
    """Short text for titles/stats, and the tag used in output file names."""
    bc, group, incl = bias_settings(cfg)
    if bc:
        how = "hofx" if incl else f"hofx + {group}"
        return f"model BT bias-corrected ({how})", "BC"
    how = f"hofx − {group}" if incl else "hofx"
    return f"model BT NOT bias-corrected ({how})", "noBC"


DEFAULT_FIXED_BCLR = {8: 234.0, 9: 246.0, 10: 258.8, 13: 292.0}   # K (btlim of SymmCldImpact)


def clr_settings(cfg):
    """(method, {ch: Bclr}) for the clear-sky BT in CA:
    clear_sky_hofx -> Bclr = <clear_sky_group>/brightnessTemperature from the file
    fixed          -> Bclr = one fixed value per channel (ca.fixed_bclr)"""
    c = cfg.get("ca", {}) or {}
    method = str(c.get("method", "clear_sky_hofx")).lower()
    if method not in ("clear_sky_hofx", "fixed"):
        raise ValueError(f"ca.method must be clear_sky_hofx or fixed, not {method}")
    fixed = c.get("fixed_bclr") or DEFAULT_FIXED_BCLR
    return method, {int(k): float(v) for k, v in fixed.items()}


def clr_text(cfg, channels):
    """(title text, file-name suffix) describing Bclr"""
    method, fixed = clr_settings(cfg)
    if method == "fixed":
        vals = ", ".join(f"ch{ch} {fixed[ch]:g}" for ch in channels if ch in fixed)
        return f"Bclr fixed ({vals} K)", "-fixedBclr"
    return f"Bclr = {cfg.get('clear_sky_group', 'ClearSkyHofX')}", ""


def read_file(path, channels, qc_value, cfg, need_ca=False):
    """Return {ch: {RAW_KEYS: array}} for every point with valid obs and model BT in
    one file (before QC). qc_pass = EffectiveQC == qc_value and a valid
    EffectiveError > 0.

    Model BT ("hofx") is bias-corrected or not following `bias_correction`;
    `bias.hofx_includes_bias` says whether the hofx group already contains the bias
    (JEDI default: hofx = H(x) + ObsBias).

    need_ca -> also the symmetric cloud impact (Okamoto et al. 2014, 2017 Eq. 1)
        CA = (|B - Bclr| + |O - Bclr|) / 2
    always WITHOUT bias correction: B = H(x) without bias, Bclr = clear-sky H(x)
    (`clear_sky_group`, default ClearSkyHofX; ca.method: clear_sky_hofx) or a fixed
    value per channel (ca.method: fixed). CA = NaN where Bclr is missing."""
    bc, bgroup, incl = bias_settings(cfg)
    clr_group = cfg.get("clear_sky_group", "ClearSkyHofX")
    clr_method, clr_fixed = clr_settings(cfg)
    with h5py.File(path, "r") as f:
        chan_list = np.asarray(f["Channel"][:]).astype(int)
        obs = np.asarray(f["ObsValue/" + GROUP_VAR][:], dtype=np.float64)
        hofx = np.asarray(f["hofx/" + GROUP_VAR][:], dtype=np.float64)
        err = np.asarray(f["EffectiveError/" + GROUP_VAR][:], dtype=np.float64)
        qc = np.asarray(f["EffectiveQC/" + GROUP_VAR][:])
        bkey = f"{bgroup}/{GROUP_VAR}"
        if bkey in f:
            bias = np.asarray(f[bkey][:], dtype=np.float64)
        else:
            bias = None
            if bc and not incl:
                raise KeyError(f"bias_correction: true needs {bkey} in {path} "
                               f"(or set bias_correction: false)")
            if (bc or incl) and not getattr(read_file, "_warned", False):
                print(f"WARNING: no {bkey} in {path}: taking hofx as it is "
                      f"(bias = 0) for both bias-corrected and raw model BT")
                read_file._warned = True
        clr = None
        if need_ca and clr_method == "clear_sky_hofx":
            ckey = f"{clr_group}/{GROUP_VAR}"
            if ckey not in f:
                raise KeyError(f"{ckey} (clear-sky BT for CA) not in {path}")
            clr = np.asarray(f[ckey][:], dtype=np.float64)

    out = {}
    for ch in channels:
        idx = np.where(chan_list == ch)[0]
        if idx.size == 0:
            raise KeyError(f"channel {ch} not in file {path} (has {chan_list.tolist()})")
        i = idx[0]
        o, h, e, q = obs[:, i], hofx[:, i], err[:, i], qc[:, i]
        if bias is None:
            b = np.zeros_like(h)
        else:
            b = np.where(_ok(bias[:, i]), bias[:, i], np.nan)
        h_raw = h - b if incl else h            # H(x) without bias correction
        h_bc = h if incl else h + b             # bias-corrected H(x)
        hm = h_bc if bc else h_raw
        valid = _ok(o) & _ok(h) & np.isfinite(hm)
        e_ok = _ok(e) & (e > 0.0)
        d = {"obs": o[valid], "hofx": hm[valid], "err": e[valid],
             "qc_pass": ((q == qc_value) & e_ok)[valid]}
        if need_ca:
            if clr is None:                         # ca.method: fixed
                c = np.full_like(o, clr_fixed[ch])
            else:
                c = clr[:, i]
                c = np.where(_ok(c), c, np.nan)
            ca = 0.5 * (np.abs(h_raw - c) + np.abs(o - c))
            d["ca"] = ca[valid]
        out[ch] = d
    return out


RAW_KEYS = ("obs", "hofx", "err", "qc_pass")


def qc_view(raw):
    """{ch: raw arrays} -> {ch: {omb, nomb, obs, hofx[, ca]}} for QC-passed points only."""
    out = {}
    for ch, d in raw.items():
        ok = d["qc_pass"]
        o, h = d["obs"][ok], d["hofx"][ok]
        omb = o - h
        out[ch] = {"omb": omb, "nomb": omb / d["err"][ok], "obs": o, "hofx": h}
        if "ca" in d:
            out[ch]["ca"] = d["ca"][ok]
    return out


def collect(cfg, exp, cycles, channels, need_ca=False):
    """Read all cycles of one experiment; returns ({ch: {RAW_KEYS: array}}, used cycles)
    with every valid point (QC not applied yet, see qc_view)."""
    templates = dict(cfg.get("templates", {}) or {})
    keys = RAW_KEYS + (("ca",) if need_ca else ())
    data = {ch: {k: [] for k in keys} for ch in channels}
    used, missing = [], []
    for cyc in cycles:
        path = cfg["path_template"].format(
            root=cfg.get("root", ""), experiment=exp["name"], cycle=cyc,
            lead=cfg.get("lead", 0), **templates)
        if not os.path.isfile(path):
            if cfg.get("skip_missing", True):
                missing.append(cyc)
                continue
            raise FileNotFoundError(path)
        res = read_file(path, channels, int(cfg.get("qc_value", 0)), cfg, need_ca)
        for ch, d in res.items():
            for k in keys:
                data[ch][k].append(d[k])
        used.append(cyc)
    for ch in channels:
        for k in keys:
            data[ch][k] = (np.concatenate(data[ch][k]) if data[ch][k]
                           else np.empty(0, dtype=bool if k == "qc_pass" else np.float64))
    print(f"[{exp['label']}] cycles used: {len(used)}, missing: {len(missing)}")
    if missing:
        print(f"[{exp['label']}]   missing: {' '.join(missing)}")
    return data, used


# ---------------------------------------------------------------------------
# Stats / plotting
# ---------------------------------------------------------------------------
def stats(x):
    if x.size == 0:
        return dict(n=0, mean=np.nan, std=np.nan, rms=np.nan)
    return dict(n=int(x.size), mean=float(np.mean(x)), std=float(np.std(x)),
                rms=float(np.sqrt(np.mean(x * x))))


def plot_kind(kind, all_data, exps, channels, cfg, ncycles):
    pcfg = cfg.get("plot", {}) or {}
    kcfg = pcfg.get("omb" if kind == "omb" else "normalized_omb", {}) or {}
    # hist_range: range used for binning; xlim: displayed x-axis range.
    # ("range" kept as a backward-compatible alias of hist_range.)
    default_rng = [-20.0, 20.0] if kind == "omb" else [-6.0, 6.0]
    lo, hi = kcfg.get("hist_range", kcfg.get("range", default_rng))
    xlo, xhi = kcfg.get("xlim", [lo, hi])
    clip = bool(kcfg.get("clip_to_edges", True))
    bw = float(kcfg.get("bin_width", 0.5 if kind == "omb" else 0.2))
    bins = np.arange(lo, hi + 0.5 * bw, bw)
    density = bool(pcfg.get("density", True))
    log_y = bool(pcfg.get("log_y", False))
    stats_loc = pcfg.get("stats_loc", "lower center")
    style = pcfg.get("style", "line")   # line | hist
    ncols = int(pcfg.get("ncols", 2))
    nrows = int(np.ceil(len(channels) / ncols))
    figsize = pcfg.get("figsize", [5.5 * ncols, 4.0 * nrows])

    xlabel = "OmB (K)" if kind == "omb" else "OmB / EffectiveError"
    title = "OmB" if kind == "omb" else "Normalized OmB"

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ip, ch in enumerate(channels):
        ax = axes[ip // ncols][ip % ncols]
        lines = []
        for ie, exp in enumerate(exps):
            x = all_data[exp["label"]][ch][kind]
            s = stats(x)
            c = colors[ie % len(colors)]
            if x.size:
                xx = np.clip(x, lo, hi) if clip else x
                if style == "line":
                    # bin counts drawn as a line through bin centers
                    cnt, edges = np.histogram(xx, bins=bins, density=density)
                    ax.plot(0.5 * (edges[:-1] + edges[1:]), cnt,
                            color=c, lw=1.5, label=exp["label"])
                else:
                    ax.hist(xx, bins=bins, density=density,
                            histtype="step" if len(exps) > 1 else "stepfilled",
                            color=c, alpha=1.0 if len(exps) > 1 else 0.6,
                            lw=1.5, label=exp["label"])
            lines.append(f"{exp['label']}: N={s['n']:,}  mean={s['mean']:.3f}  "
                         f"std={s['std']:.3f}  rms={s['rms']:.3f}")
        if kind == "nomb" and kcfg.get("gaussian", True) and density:
            xg = np.linspace(min(lo, xlo), max(hi, xhi), 400)
            ax.plot(xg, np.exp(-0.5 * xg ** 2) / np.sqrt(2 * np.pi),
                    "k--", lw=1.0, label="N(0,1)")
        ax.axvline(0.0, color="gray", lw=0.8)
        ax.set_xlim(xlo, xhi)
        if log_y:
            ax.set_yscale("log")
        ax.set_title(f"Channel {ch}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("PDF" if density else "Count")
        # Stats box inside the panel, along the bottom edge
        sx, sha = {"lower left": (0.02, "left"), "lower center": (0.5, "center"),
                   "lower right": (0.98, "right")}[stats_loc]
        ax.text(sx, 0.03, "\n".join(lines), transform=ax.transAxes,
                va="bottom", ha=sha, fontsize=7.5, zorder=5,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
        if len(exps) > 1 or kind == "nomb":
            ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    for ip in range(len(channels), nrows * ncols):
        axes[ip // ncols][ip % ncols].set_visible(False)

    qc = cfg.get("qc_value", 0)
    sub = (f"histogram range [{lo}, {hi}]"
           + (", outside values clipped into edge bins" if clip else ", outside values dropped"))
    fig.suptitle(f"AHI Himawari-9 {title}  (EffectiveQC=={qc}, {ncycles} cycles, "
                 f"lead {cfg.get('lead', 0)}h)\n{bias_text(cfg)[0]};  {sub}", fontsize=11)
    fig.tight_layout()
    return fig


def scatter_range(scfg, ch, all_data, exps):
    """Axis range for one channel: plot.scatter.range is [lo, hi] for every channel,
    a {channel: [lo, hi]} dict, or null -> 0.1-99.9 percentiles of obs and model BT
    over all experiments (rounded out to 5 K)."""
    rng = scfg.get("range")
    if isinstance(rng, dict):
        rng = rng.get(ch, rng.get(str(ch)))
    if rng is not None:
        return float(rng[0]), float(rng[1])
    vals = [all_data[e["label"]][ch][k] for e in exps for k in ("obs", "hofx")]
    vals = np.concatenate(vals) if vals else np.empty(0)
    if vals.size == 0:
        return 180.0, 320.0
    lo, hi = np.percentile(vals, [0.1, 99.9])
    return 5.0 * np.floor(lo / 5.0), 5.0 * np.ceil(hi / 5.0)


# Discrete jet-like palette after Okamoto (2017, QJRMS 143:1517, Figs 2-3):
# dark navy -> blue -> light blue -> cyan -> green -> yellow -> orange -> red -> dark red
OKAMOTO_COLORS = [
    "#1b1d5e", "#262f86", "#2e45a8", "#3560c4", "#3d7ad6", "#4b94df",
    "#5aaee0", "#6cc5d4", "#86d2a6", "#a8d86c", "#cbdc3c", "#ecd82a",
    "#f6ad25", "#f07b20", "#e2441c", "#b51d1d", "#7a1010",
]


def scatter_cmap(name):
    if name in (None, "okamoto"):
        return LinearSegmentedColormap.from_list("okamoto", OKAMOTO_COLORS)
    return plt.get_cmap(name)


def density_norm(scfg, vmax, vmin_pos, normalize, cmap, step):
    """Discrete log10 colour levels, whole decades at both ends (0..4 in Okamoto).
    vmax / vmin_pos: largest / smallest non-empty bin over all panels."""
    vmin = scfg.get("vmin", vmin_pos if normalize else 1.0)
    vmax = scfg.get("vmax", vmax)
    if not np.isfinite(vmin) or vmax <= 0:
        vmin, vmax = 1.0, 10.0
    llo, lhi = np.floor(np.log10(vmin)), np.ceil(np.log10(vmax))
    if lhi <= llo:
        lhi = llo + 1.0
    levels = np.arange(llo, lhi + 0.5 * step, step)
    return llo, lhi, BoundaryNorm(levels, ncolors=cmap.N, clip=True)


def _density_colorbar(fig, mesh, axes, ncols, llo, lhi, normalize):
    cb = fig.colorbar(mesh, ax=axes, orientation="horizontal",
                      shrink=min(0.6, 0.25 * ncols + 0.1), aspect=35, pad=0.02,
                      ticks=np.arange(llo, lhi + 0.5, 1.0))
    cb.set_label("log10(fraction of points per bin)" if normalize
                 else "log10(number per bin)")
    cb.ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%g"))
    return cb


def plot_cloud_impact(data, exps, cfg, ncycles):
    """Density plot of the symmetric cloud impact CA (x) and O-B (y) after Okamoto
    (2017, QJRMS 143:1517, Figs 3-4); colour = log10(number per bin).
    CA = (|B - Bclr| + |O - Bclr|) / 2 is computed without bias correction; O-B
    follows `bias_correction`. Rows = experiments, columns = channels.
    `data` is {label: {ch: {obs, hofx, ca}}}, QC-selected or not (apply_qc)."""
    pcfg = cfg.get("plot", {}) or {}
    ccfg = pcfg.get("cloud_impact", {}) or {}
    channels = [int(c) for c in ccfg.get("channels", [8, 9, 10, 13])]
    apply_qc = bool(ccfg.get("apply_qc", True))
    bw = float(ccfg.get("bin_width", 1.0))
    xlo, xhi = [float(v) for v in ccfg.get("ca_range", [0.0, 70.0])]
    ylo, yhi = [float(v) for v in ccfg.get("omb_range", [-50.0, 50.0])]
    normalize = bool(ccfg.get("normalize", False))
    cmap = scatter_cmap(ccfg.get("cmap", "okamoto"))
    step = float(ccfg.get("level_step", 0.25))
    bust = bool(ccfg.get("bust_lines", False))
    bfac = float(ccfg.get("bust_factor", 1.8))
    bca = float(ccfg.get("bust_ca_min", 3.0))
    show_stats = bool(ccfg.get("show_stats", True))
    nrows, ncols = len(exps), len(channels)
    figsize = ccfg.get("figsize", [3.6 * ncols + 0.4, 3.0 * nrows + 1.5])

    xe = np.arange(xlo, xhi + 0.5 * bw, bw)
    ye = np.arange(ylo, yhi + 0.5 * bw, bw)
    panels, vmax, vmin_pos = {}, 0.0, np.inf
    for exp in exps:
        for ch in channels:
            d = data[exp["label"]][ch]
            ok = np.isfinite(d["ca"])
            ca, omb = d["ca"][ok], (d["obs"] - d["hofx"])[ok]
            H, _, _ = np.histogram2d(ca, omb, bins=[xe, ye])
            if normalize and ca.size:
                H = H / ca.size
            H[H == 0] = np.nan
            if np.isfinite(H).any():
                vmax = max(vmax, np.nanmax(H))
                vmin_pos = min(vmin_pos, np.nanmin(H))
            panels[(exp["label"], ch)] = (H, ca, omb, int((~ok).sum()))
    llo, lhi, norm = density_norm(ccfg, vmax, vmin_pos, normalize, cmap, step)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False,
                             constrained_layout=True)
    mesh = None
    letters = "abcdefghijklmnopqrstuvwxyz"
    for ie, exp in enumerate(exps):
        for ic, ch in enumerate(channels):
            ax = axes[ie][ic]
            H, ca, omb, nmiss = panels[(exp["label"], ch)]
            with np.errstate(divide="ignore", invalid="ignore"):
                logH = np.log10(H)
            mesh = ax.pcolormesh(xe, ye, logH.T, cmap=cmap, norm=norm, rasterized=True)
            ax.axhline(0.0, ls=(0, (4, 3)), color="0.35", lw=0.7, zorder=3)
            if bust:                               # |O-B| = bfac * CA for CA > bca
                xx = np.array([bca, xhi])
                for s in (1, -1):
                    ax.plot(xx, s * bfac * xx, ls=(0, (8, 4)), color="0.2", lw=0.9,
                            zorder=3)
            ax.set_xlim(xlo, xhi)
            ax.set_ylim(ylo, yhi)
            if show_stats:
                if ca.size:
                    txt = (f"N={ca.size:,}\nO-B mean={np.mean(omb):.2f}\n"
                           f"O-B SD={np.std(omb):.2f}\nCA mean={np.mean(ca):.2f}")
                else:
                    txt = "N=0"
                if nmiss:
                    txt += f"\n(no Bclr: {nmiss:,})"
                ax.text(0.97, 0.03, txt, transform=ax.transAxes, va="bottom", ha="right",
                        fontsize=7, zorder=4,
                        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8))
            k = ie * ncols + ic
            ax.set_title(f"({letters[k % 26]})  {exp['label']}  Band{ch}",
                         loc="left", fontsize=10)
            ax.set_xlabel("CA (K)" if ie == nrows - 1 else "")
            ax.set_ylabel("O−B (K)" if ic == 0 else "")
            ax.tick_params(direction="out")

    qc = (f"EffectiveQC=={cfg.get('qc_value', 0)}" if apply_qc
          else "before QC (all valid obs)")
    extra = (f";  long dashed: |O−B| = {bfac:g} CA for CA > {bca:g} K" if bust else "")
    fig.suptitle(f"AHI Himawari-9 symmetric cloud impact CA vs O−B  ({qc}, {ncycles} "
                 f"cycles, lead {cfg.get('lead', 0)}h)\n"
                 f"O−B: {bias_text(cfg)[0]};  CA = (|B−Bclr|+|O−Bclr|)/2 without bias "
                 f"correction, {clr_text(cfg, channels)[0]}\ncolour: log10 number in {bw:g} K bins{extra}",
                 fontsize=10.5)
    _density_colorbar(fig, mesh, axes, ncols, llo, lhi, normalize)
    return fig


def plot_scatter(all_data, exps, cfg, ncycles):
    """Density scatter plot after Okamoto (2017, QJRMS, Fig. 2): model BT (x) vs
    observed BT (y) as a 2-D histogram, coloured by log10(count per bin) in discrete
    steps. Dashed lines mark O-B = const (plot.scatter.omb_lines).
    Rows = experiments, columns = channels; one colour scale shared by all panels
    so the experiments can be compared directly."""
    pcfg = cfg.get("plot", {}) or {}
    scfg = pcfg.get("scatter", {}) or {}
    channels = [int(c) for c in scfg.get("channels", [8, 9, 10, 13])]
    bw = float(scfg.get("bin_width", 0.5))
    normalize = bool(scfg.get("normalize", False))   # True -> fraction of points in panel
    cmap = scatter_cmap(scfg.get("cmap", "okamoto"))
    step = float(scfg.get("level_step", 0.25))       # colour step in log10 units
    omb_lines = [float(x) for x in scfg.get("omb_lines", [0, 10, -10, 30, -30])]
    show_stats = bool(scfg.get("show_stats", True))
    nrows, ncols = len(exps), len(channels)
    figsize = scfg.get("figsize", [3.5 * ncols + 0.4, 3.5 * nrows + 1.3])

    # 2-D histograms first, so every panel can share one colour scale
    panels, vmax, vmin_pos = {}, 0.0, np.inf
    for ch in channels:
        lo, hi = scatter_range(scfg, ch, all_data, exps)
        edges = np.arange(lo, hi + 0.5 * bw, bw)
        for exp in exps:
            d = all_data[exp["label"]][ch]
            h, o = d["hofx"], d["obs"]
            H, _, _ = np.histogram2d(h, o, bins=[edges, edges])
            if normalize and h.size:
                H = H / h.size
            H[H == 0] = np.nan                    # empty bins left blank
            if np.isfinite(H).any():
                vmax = max(vmax, np.nanmax(H))
                vmin_pos = min(vmin_pos, np.nanmin(H))
            panels[(exp["label"], ch)] = (H, edges, h, o)
    llo, lhi, norm = density_norm(scfg, vmax, vmin_pos, normalize, cmap, step)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False,
                             constrained_layout=True)
    mesh = None
    letters = "abcdefghijklmnopqrstuvwxyz"
    for ie, exp in enumerate(exps):
        for ic, ch in enumerate(channels):
            ax = axes[ie][ic]
            H, edges, h, o = panels[(exp["label"], ch)]
            with np.errstate(divide="ignore", invalid="ignore"):
                logH = np.log10(H)
            mesh = ax.pcolormesh(edges, edges, logH.T, cmap=cmap, norm=norm,
                                 rasterized=True)
            lo, hi = edges[0], edges[-1]
            for d in omb_lines:                   # O - B = d  ->  y = x + d
                ax.plot([lo, hi], [lo + d, hi + d], ls=(0, (4, 3)), color="0.35",
                        lw=0.7, zorder=3)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            if show_stats:
                if h.size:
                    omb = o - h
                    r = np.corrcoef(h, o)[0, 1] if h.size > 1 else np.nan
                    txt = (f"N={h.size:,}\nO-B mean={np.mean(omb):.2f}\n"
                           f"O-B SD={np.std(omb):.2f}\nR={r:.3f}")
                else:
                    txt = "N=0"
                ax.text(0.97, 0.03, txt, transform=ax.transAxes, va="bottom", ha="right",
                        fontsize=7, zorder=4,
                        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8))
            k = ie * ncols + ic
            ax.set_title(f"({letters[k % 26]})  {exp['label']}  Band{ch}",
                         loc="left", fontsize=10)
            ax.set_xlabel("model (K)" if ie == nrows - 1 else "")
            ax.set_ylabel("observation (K)" if ic == 0 else "")
            ax.tick_params(direction="out")

    qc = cfg.get("qc_value", 0)
    lines = ", ".join(f"{d:+g}" if d else "0" for d in omb_lines)
    fig.suptitle(f"AHI Himawari-9 density scatter of model and observed BT  "
                 f"(EffectiveQC=={qc}, {ncycles} cycles, lead {cfg.get('lead', 0)}h)\n"
                 f"{bias_text(cfg)[0]};  colour: log10 number in {bw:g} K bins;  "
                 f"dashed: O−B = {lines} K", fontsize=10.5)
    _density_colorbar(fig, mesh, axes, ncols, llo, lhi, normalize)
    return fig


def _box_label(name, x, show_stats):
    if not show_stats or x.size == 0:
        return name
    return f"{name}\nmean {np.mean(x):.1f}\nSD {np.std(x):.1f}"


def _draw_boxes(ax, arrays, labels, colors, bcfg):
    """Boxes = Q1..Q3, whiskers = whis * IQR, dots = outliers,
    thick line = median, x = mean (as in Okamoto 2017, QJRMS, Fig. 11)."""
    arrays = [a if a.size else np.array([np.nan]) for a in arrays]
    bp = ax.boxplot(
        arrays, whis=float(bcfg.get("whis", 1.5)), widths=0.6,
        showmeans=True, showfliers=bool(bcfg.get("show_outliers", True)),
        patch_artist=True,
        medianprops=dict(color="k", lw=2.0),
        meanprops=dict(marker="x", markeredgecolor="k", markersize=7),
        flierprops=dict(marker=".", markersize=2, markeredgecolor="0.35",
                        markerfacecolor="0.35", alpha=0.4))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
        patch.set_edgecolor("k")
    for fl in bp["fliers"]:
        fl.set_rasterized(True)        # many outliers -> keep vector output small
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def plot_boxplot(data, exps, cfg, ncycles):
    """Okamoto (2017, QJRMS, Fig. 11)-style boxplots, one row per channel:
         left : observed BT and model BT of every experiment
         right: O-B of every experiment
    `data` is {label: {ch: {obs, hofx}}}, already QC-selected or not (apply_qc)."""
    pcfg = cfg.get("plot", {}) or {}
    bcfg = pcfg.get("boxplot", {}) or {}
    channels = [int(c) for c in bcfg.get("channels", [8, 9, 10, 13])]
    apply_qc = bool(bcfg.get("apply_qc", True))
    obs_mode = bcfg.get("obs", "auto")        # auto | shared | per_experiment
    show_stats = bool(bcfg.get("show_stats", True))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    obs_color = "0.6"

    def shared_obs(ch):
        if obs_mode == "shared":
            return True
        if obs_mode == "per_experiment":
            return False
        o0 = data[exps[0]["label"]][ch]["obs"]
        return all(np.array_equal(o0, data[e["label"]][ch]["obs"]) for e in exps[1:])

    n_a = len(exps) + (1 if all(shared_obs(ch) for ch in channels) else len(exps))
    nrows = len(channels)
    figsize = bcfg.get("figsize", [1.3 * (n_a + len(exps)) + 2.5, 3.8 * nrows + 0.6])
    fig, axes = plt.subplots(nrows, 2, figsize=figsize, squeeze=False,
                             gridspec_kw=dict(width_ratios=[n_a, len(exps) + 0.5]),
                             constrained_layout=True)

    letters = "abcdefghijklmnopqrstuvwxyz"
    for ir, ch in enumerate(channels):
        d = {e["label"]: data[e["label"]][ch] for e in exps}
        # (a) observation / model BT
        if shared_obs(ch):
            arrs = [d[exps[0]["label"]]["obs"]]
            labs = [_box_label("OB", arrs[0], show_stats)]
            cols = [obs_color]
        else:
            arrs = [d[e["label"]]["obs"] for e in exps]
            labs = [_box_label(f"{e['label']} OB", a, show_stats) for e, a in zip(exps, arrs)]
            cols = [obs_color] * len(exps)
        for ie, e in enumerate(exps):
            h = d[e["label"]]["hofx"]
            arrs.append(h)
            labs.append(_box_label(e["label"], h, show_stats))
            cols.append(colors[ie % len(colors)])
        ax = axes[ir][0]
        _draw_boxes(ax, arrs, labs, cols, bcfg)
        if bcfg.get("bt_ylim"):
            ax.set_ylim(*bcfg["bt_ylim"])
        ax.set_ylabel("Observation / Model BT (K)")
        ax.set_title(f"({letters[(2 * ir) % 26]}) Observation / Model BT  (Ch {ch})",
                     loc="left", fontsize=10)

        # (b) O-B
        ombs = [d[e["label"]]["obs"] - d[e["label"]]["hofx"] for e in exps]
        ax = axes[ir][1]
        _draw_boxes(ax, ombs,
                    [_box_label(e["label"], a, show_stats) for e, a in zip(exps, ombs)],
                    [colors[ie % len(colors)] for ie in range(len(exps))], bcfg)
        ax.axhline(0.0, color="gray", lw=0.8, zorder=0)
        if bcfg.get("omb_ylim"):
            ax.set_ylim(*bcfg["omb_ylim"])
        ax.set_ylabel("O−B (K)")
        ax.set_title(f"({letters[(2 * ir + 1) % 26]}) O−B  (Ch {ch})",
                     loc="left", fontsize=10)

    qc = (f"EffectiveQC=={cfg.get('qc_value', 0)}" if apply_qc
          else "before QC (all valid obs)")
    fig.suptitle(f"AHI Himawari-9 BT and O−B distributions  ({qc}, {ncycles} cycles, "
                 f"lead {cfg.get('lead', 0)}h)\n{bias_text(cfg)[0]}\n"
                 f"box: Q1–Q3, whiskers: {float(bcfg.get('whis', 1.5)):g}×IQR, "
                 f"line: median, ×: mean", fontsize=11)
    return fig


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    with open(sys.argv[1]) as f:
        cfg = yaml.safe_load(f)

    channels = [int(c) for c in cfg.get("channels", [8, 9, 10, 13])]
    pcfg = cfg.get("plot", {}) or {}
    scfg = pcfg.get("scatter", {}) or {}
    bcfg = pcfg.get("boxplot", {}) or {}
    ccfg = pcfg.get("cloud_impact", {}) or {}
    do_scatter = bool(scfg.get("enabled", False))
    do_box = bool(bcfg.get("enabled", False))
    do_ca = bool(ccfg.get("enabled", False))
    # boxplot / cloud-impact plot may need pre-QC data
    box_raw = do_box and not bool(bcfg.get("apply_qc", True))
    ca_raw = do_ca and not bool(ccfg.get("apply_qc", True))
    read_channels = list(channels)
    for on, sub in ((do_scatter, scfg), (do_box, bcfg), (do_ca, ccfg)):
        if on:
            read_channels += [int(c) for c in sub.get("channels", [8, 9, 10, 13])
                              if int(c) not in read_channels]
    cycles = build_cycles(cfg["cycles"])
    exps = normalize_experiments(cfg)
    btxt, btag = bias_text(cfg)
    print(f"cycles requested: {len(cycles)} ({cycles[0]} -> {cycles[-1]})")
    print(btxt)
    if do_ca:
        method, fixed = clr_settings(cfg)
        ca_ch = [int(c) for c in ccfg.get("channels", [8, 9, 10, 13])]
        miss = [ch for ch in ca_ch if ch not in fixed]
        if method == "fixed" and miss:
            raise KeyError(f"ca.fixed_bclr has no value for channel(s) {miss}")
        print("CA: " + clr_text(cfg, ca_ch)[0])

    all_data, raw_data, ncyc = {}, {}, 0
    for exp in exps:
        raw, used = collect(cfg, exp, cycles, read_channels, need_ca=do_ca)
        all_data[exp["label"]] = qc_view(raw)
        if box_raw or ca_raw:
            keep = ("obs", "hofx") + (("ca",) if ca_raw else ())
            raw_data[exp["label"]] = {ch: {k: d[k] for k in keep}
                                      for ch, d in raw.items()}
        del raw
        ncyc = max(ncyc, len(used))

    ocfg = cfg.get("output", {}) or {}
    odir = ocfg.get("dir", "./figs")
    os.makedirs(odir, exist_ok=True)
    prefix = ocfg.get("prefix", "ahi")
    fmt = ocfg.get("format", "png")
    dpi = int(ocfg.get("dpi", 150))
    # BC / noBC in every file name
    tag = f"{btag}_{cycles[0]}-{cycles[-1]}_{cfg.get('lead', 0)}hr"

    for kind, name in (("omb", "OmB"), ("nomb", "NormOmB")):
        fig = plot_kind(kind, all_data, exps, channels, cfg, ncyc)
        out = os.path.join(odir, f"{prefix}_{name}_{tag}.{fmt}")
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        print(f"saved {out}")

    if do_scatter:
        fig = plot_scatter(all_data, exps, cfg, ncyc)
        out = os.path.join(odir, f"{prefix}_scatter_{tag}.{fmt}")
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        print(f"saved {out}")

    if do_box:
        fig = plot_boxplot(raw_data if box_raw else all_data, exps, cfg, ncyc)
        out = os.path.join(odir, f"{prefix}_boxplot_{tag}.{fmt}")
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        print(f"saved {out}")

    if do_ca:
        fig = plot_cloud_impact(raw_data if ca_raw else all_data, exps, cfg, ncyc)
        ctag = clr_text(cfg, ccfg.get("channels", [8, 9, 10, 13]))[1]
        out = os.path.join(odir, f"{prefix}_CA{ctag}-OmB_{tag}.{fmt}")
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        print(f"saved {out}")

    if ocfg.get("stats_file", True):
        out = os.path.join(odir, f"{prefix}_stats_{tag}.txt")
        with open(out, "w") as f:
            f.write(f"# cycles {cycles[0]} -> {cycles[-1]}, lead {cfg.get('lead', 0)}h, "
                    f"EffectiveQC=={cfg.get('qc_value', 0)}\n# {btxt}\n")
            f.write(f"{'experiment':<20s}{'ch':>4s}{'kind':>8s}{'N':>12s}"
                    f"{'mean':>10s}{'std':>10s}{'rms':>10s}\n")
            for exp in exps:
                for ch in channels:
                    for kind in ("omb", "nomb"):
                        s = stats(all_data[exp["label"]][ch][kind])
                        f.write(f"{exp['label']:<20s}{ch:>4d}{kind:>8s}{s['n']:>12d}"
                                f"{s['mean']:>10.4f}{s['std']:>10.4f}{s['rms']:>10.4f}\n")
        print(f"saved {out}")


if __name__ == "__main__":
    main()
