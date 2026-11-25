#!/usr/bin/env python
"""
Multi-cycle OmB / normalized-OmB distribution for Himawari AHI channels.

  OmB            = ObsValue/brightnessTemperature - hofx/brightnessTemperature
  normalized OmB = OmB / EffectiveError/brightnessTemperature
  selection      : EffectiveQC/brightnessTemperature == qc_value (default 0)

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
def read_file(path, channels, qc_value):
    """Return {ch: (omb, nomb)} for QC-passed, valid points in one file."""
    with h5py.File(path, "r") as f:
        chan_list = np.asarray(f["Channel"][:]).astype(int)
        obs = np.asarray(f["ObsValue/" + GROUP_VAR][:], dtype=np.float64)
        hofx = np.asarray(f["hofx/" + GROUP_VAR][:], dtype=np.float64)
        err = np.asarray(f["EffectiveError/" + GROUP_VAR][:], dtype=np.float64)
        qc = np.asarray(f["EffectiveQC/" + GROUP_VAR][:])

    out = {}
    for ch in channels:
        idx = np.where(chan_list == ch)[0]
        if idx.size == 0:
            raise KeyError(f"channel {ch} not in file {path} (has {chan_list.tolist()})")
        i = idx[0]
        o, h, e, q = obs[:, i], hofx[:, i], err[:, i], qc[:, i]
        ok = (
            (q == qc_value)
            & np.isfinite(o) & np.isfinite(h) & np.isfinite(e)
            & (np.abs(o) < FILL_THRESH) & (np.abs(h) < FILL_THRESH)
            & (np.abs(e) < FILL_THRESH) & (e > 0.0)
        )
        omb = o[ok] - h[ok]
        out[ch] = (omb, omb / e[ok])
    return out


def collect(cfg, exp, cycles, channels):
    templates = dict(cfg.get("templates", {}) or {})
    data = {ch: {"omb": [], "nomb": []} for ch in channels}
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
        res = read_file(path, channels, int(cfg.get("qc_value", 0)))
        for ch, (omb, nomb) in res.items():
            data[ch]["omb"].append(omb)
            data[ch]["nomb"].append(nomb)
        used.append(cyc)
    for ch in channels:
        for k in ("omb", "nomb"):
            data[ch][k] = np.concatenate(data[ch][k]) if data[ch][k] else np.empty(0)
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
                 f"lead {cfg.get('lead', 0)}h)\n{sub}", fontsize=11)
    fig.tight_layout()
    return fig


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    with open(sys.argv[1]) as f:
        cfg = yaml.safe_load(f)

    channels = [int(c) for c in cfg.get("channels", [8, 9, 10, 13])]
    cycles = build_cycles(cfg["cycles"])
    exps = normalize_experiments(cfg)
    print(f"cycles requested: {len(cycles)} ({cycles[0]} -> {cycles[-1]})")

    all_data, ncyc = {}, 0
    for exp in exps:
        all_data[exp["label"]], used = collect(cfg, exp, cycles, channels)
        ncyc = max(ncyc, len(used))

    ocfg = cfg.get("output", {}) or {}
    odir = ocfg.get("dir", "./figs")
    os.makedirs(odir, exist_ok=True)
    prefix = ocfg.get("prefix", "ahi")
    fmt = ocfg.get("format", "png")
    dpi = int(ocfg.get("dpi", 150))
    tag = f"{cycles[0]}-{cycles[-1]}_{cfg.get('lead', 0)}hr"

    for kind, name in (("omb", "OmB"), ("nomb", "NormOmB")):
        fig = plot_kind(kind, all_data, exps, channels, cfg, ncyc)
        out = os.path.join(odir, f"{prefix}_{name}_{tag}.{fmt}")
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        print(f"saved {out}")

    if ocfg.get("stats_file", True):
        out = os.path.join(odir, f"{prefix}_stats_{tag}.txt")
        with open(out, "w") as f:
            f.write(f"# cycles {cycles[0]} -> {cycles[-1]}, lead {cfg.get('lead', 0)}h, "
                    f"EffectiveQC=={cfg.get('qc_value', 0)}\n")
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
