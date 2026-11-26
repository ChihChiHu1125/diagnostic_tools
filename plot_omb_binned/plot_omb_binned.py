#!/usr/bin/env python
"""
O-B statistics (SD, mean, ...) as a function of a binned x variable, in the style of
Okamoto et al. (2017, QJRMS 143:1517, Fig. 5): one curve per statistic and per sample
filter on the left axis, and the log10 number of samples per bin as grey bars on the
right axis. Default x variable: symmetric cloud impact
    CA = (|B - Bclr| + |O - Bclr|) / 2           (Okamoto et al. 2014)

Inputs: IODA obsout files (ObsValue, hofx, EffectiveQC, EffectiveError, ObsBias,
ClearSkyHofX groups of brightnessTemperature), many cycles, one or more experiments.
Statistics are accumulated bin by bin per file (count, sum, sum of squares), so the
memory use does not grow with the number of cycles.

Output (<tag> = BC|noBC_<first cycle>-<last cycle>_<lead>hr):
  <prefix>_OmB-vs-<xvar>_<tag>.<fmt>   rows = experiments, columns = channels
  <prefix>_OmB-vs-<xvar>_<tag>.txt     the binned numbers behind the figure

Extending:
  * a new x variable / filter variable -> add a function with @register_var
  * a new sample filter type           -> add a function with @register_filter
  * a new statistic                    -> add a function with @register_stat
  The yaml refers to them by the registered name.

Usage:
  python plot_omb_binned.py plot_omb_binned.yaml
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
# Registries
# ---------------------------------------------------------------------------
VARS = {}      # name -> (func(d) -> array, label, needs)
FILTERS = {}   # type -> func(d, spec) -> bool mask
STATS = {}     # name -> func(n, s, ss) -> array


def register_var(name, label, needs=()):
    """Variable computed from one channel's arrays d (see read_file for the keys).
    `needs` lists optional inputs that must be read for it (e.g. "clr")."""
    def deco(f):
        VARS[name] = (f, label, tuple(needs))
        return f
    return deco


def register_filter(kind):
    """Sample filter: f(d, spec) -> boolean mask (True = keep). `spec` is one entry of
    `conditions` in the yaml, e.g. {type: qc, value: 0}."""
    def deco(f):
        FILTERS[kind] = f
        return f
    return deco


def register_stat(name):
    """Statistic of O-B from per-bin count n, sum s and sum of squares ss."""
    def deco(f):
        STATS[name] = f
        return f
    return deco


# ---- variables (all in K) ---------------------------------------------------
# d keys: obs, hofx (bias-corrected or not, following bias_correction),
#         hofx_raw (never bias-corrected), clr (clear-sky H(x), raw), err, qc
@register_var("CA", "CA (K)", needs=("clr",))
def _v_ca(d):
    """symmetric cloud impact, without bias correction"""
    return 0.5 * (np.abs(d["hofx_raw"] - d["clr"]) + np.abs(d["obs"] - d["clr"]))


@register_var("cloud_obs", "Bclr − O (K)", needs=("clr",))
def _v_cobs(d):
    return d["clr"] - d["obs"]


@register_var("cloud_model", "Bclr − B (K)", needs=("clr",))
def _v_cmod(d):
    return d["clr"] - d["hofx_raw"]


@register_var("obs", "observed BT (K)")
def _v_obs(d):
    return d["obs"]


@register_var("hofx", "model BT (K)")
def _v_hofx(d):
    return d["hofx"]


@register_var("omb", "O−B (K)")
def _v_omb(d):
    return d["obs"] - d["hofx"]


# ---- filters ----------------------------------------------------------------
@register_filter("qc")
def _f_qc(d, spec):
    """{type: qc, value: 0} -> EffectiveQC == value (value may be a list)"""
    v = spec.get("value", 0)
    return np.isin(d["qc"], np.atleast_1d(v))


@register_filter("range")
def _f_range(d, spec):
    """{type: range, var: obs, min: 230, max: 330} -> keep min <= var <= max
    (either bound optional; var = any registered variable)"""
    x = get_var(d, spec["var"])
    m = np.isfinite(x)
    if spec.get("min") is not None:
        m &= x >= float(spec["min"])
    if spec.get("max") is not None:
        m &= x <= float(spec["max"])
    return m


# ---- statistics -------------------------------------------------------------
@register_stat("mean")
def _s_mean(n, s, ss):
    return s / n


@register_stat("std")
def _s_std(n, s, ss):
    m = s / n
    return np.sqrt(np.maximum(ss / n - m * m, 0.0))


@register_stat("rms")
def _s_rms(n, s, ss):
    return np.sqrt(ss / n)


def get_var(d, name):
    if name not in VARS:
        raise KeyError(f"unknown variable '{name}' (known: {', '.join(VARS)})")
    return VARS[name][0](d)


def filter_mask(d, conditions):
    m = np.ones(d["obs"].shape, dtype=bool)
    for spec in conditions or []:
        kind = spec.get("type")
        if kind not in FILTERS:
            raise KeyError(f"unknown filter type '{kind}' (known: {', '.join(FILTERS)})")
        m &= FILTERS[kind](d, spec)
    return m


def needed_inputs(cfg):
    """optional inputs (e.g. clr) needed by the x variable and all filters"""
    names = [cfg["x"]["var"]]
    for f in (cfg.get("filters") or {}).values():
        names += [c["var"] for c in f.get("conditions") or [] if "var" in c]
    need = set()
    for n in names:
        if n not in VARS:
            raise KeyError(f"unknown variable '{n}' (known: {', '.join(VARS)})")
        need.update(VARS[n][2])
    return need


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def build_cycles(spec):
    if isinstance(spec, (list, tuple)):
        return [str(c) for c in spec]
    t0 = datetime.strptime(str(spec["start"]), "%Y%m%d%H")
    t1 = datetime.strptime(str(spec["end"]), "%Y%m%d%H")
    dt = timedelta(hours=int(spec.get("interval_hours", 6)))
    out, t = [], t0
    while t <= t1:
        out.append(t.strftime("%Y%m%d%H"))
        t += dt
    return out


def normalize_experiments(cfg):
    out = []
    for e in cfg["experiments"]:
        if isinstance(e, str):
            out.append({"name": e, "label": e})
        else:
            out.append({"name": e["name"], "label": e.get("label", e["name"])})
    return out


def bias_settings(cfg):
    b = cfg.get("bias", {}) or {}
    return (bool(cfg.get("bias_correction", True)), b.get("group", "ObsBias"),
            bool(b.get("hofx_includes_bias", True)))


def bias_text(cfg):
    bc, group, incl = bias_settings(cfg)
    if bc:
        return f"model BT bias-corrected ({'hofx' if incl else 'hofx + ' + group})", "BC"
    return f"model BT NOT bias-corrected ({'hofx − ' + group if incl else 'hofx'})", "noBC"


def per_channel(value, ch, default=None):
    """value may be one setting for all channels or a {channel: setting} dict"""
    if isinstance(value, dict):
        return value.get(ch, value.get(str(ch), default))
    return default if value is None else value


def x_edges(cfg, ch):
    xc = cfg["x"]
    lo, hi = per_channel(xc.get("range"), ch, [0.0, 60.0])
    bw = float(per_channel(xc.get("bin_width"), ch, 1.0))
    return np.arange(float(lo), float(hi) + 0.5 * bw, bw)


# ---------------------------------------------------------------------------
# Reading + accumulation
# ---------------------------------------------------------------------------
def _ok(a):
    return np.isfinite(a) & (np.abs(a) < FILL_THRESH)


def read_file(path, channels, cfg, need):
    """{ch: d} with d = obs, hofx, hofx_raw, err, qc[, clr] for every location with
    valid obs and model BT (missing clr -> NaN)."""
    bc, bgroup, incl = bias_settings(cfg)
    with h5py.File(path, "r") as f:
        chan_list = np.asarray(f["Channel"][:]).astype(int)
        obs = np.asarray(f["ObsValue/" + GROUP_VAR][:], dtype=np.float64)
        hofx = np.asarray(f["hofx/" + GROUP_VAR][:], dtype=np.float64)
        err = np.asarray(f["EffectiveError/" + GROUP_VAR][:], dtype=np.float64)
        qc = np.asarray(f["EffectiveQC/" + GROUP_VAR][:])
        bkey = f"{bgroup}/{GROUP_VAR}"
        bias = np.asarray(f[bkey][:], dtype=np.float64) if bkey in f else None
        if bias is None:
            if bc and not incl:
                raise KeyError(f"bias_correction: true needs {bkey} in {path} "
                               f"(or set bias_correction: false)")
            if not getattr(read_file, "_warned", False):
                print(f"WARNING: no {bkey} in {path}: bias taken as 0")
                read_file._warned = True
        clr = None
        if "clr" in need:
            ckey = f"{cfg.get('clear_sky_group', 'ClearSkyHofX')}/{GROUP_VAR}"
            if ckey not in f:
                raise KeyError(f"{ckey} (clear-sky BT) not in {path}")
            clr = np.asarray(f[ckey][:], dtype=np.float64)

    out = {}
    for ch in channels:
        idx = np.where(chan_list == ch)[0]
        if idx.size == 0:
            raise KeyError(f"channel {ch} not in {path} (has {chan_list.tolist()})")
        i = idx[0]
        o, h = obs[:, i], hofx[:, i]
        b = np.zeros_like(h) if bias is None else np.where(_ok(bias[:, i]), bias[:, i], np.nan)
        h_raw = h - b if incl else h
        h_bc = h if incl else h + b
        hm = h_bc if bc else h_raw
        v = _ok(o) & _ok(h) & np.isfinite(hm)
        d = {"obs": o[v], "hofx": hm[v], "hofx_raw": h_raw[v], "err": err[v, i],
             "qc": qc[v, i]}
        if clr is not None:
            c = clr[:, i]
            d["clr"] = np.where(_ok(c), c, np.nan)[v]
        out[ch] = d
    return out


def new_acc(nb):
    return {"n": np.zeros(nb), "s": np.zeros(nb), "ss": np.zeros(nb)}


def accumulate(acc, xbin, omb, mask):
    """add O-B of the masked points to the per-bin count / sum / sum of squares"""
    k = xbin[mask]
    y = omb[mask]
    nb = acc["n"].size
    acc["n"] += np.bincount(k, minlength=nb)
    acc["s"] += np.bincount(k, weights=y, minlength=nb)
    acc["ss"] += np.bincount(k, weights=y * y, minlength=nb)


def collect(cfg, exp, cycles, channels, filters, need):
    """{ch: {filter: acc}} for one experiment over all cycles"""
    templates = dict(cfg.get("templates", {}) or {})
    xvar = cfg["x"]["var"]
    edges = {ch: x_edges(cfg, ch) for ch in channels}
    acc = {ch: {fn: new_acc(edges[ch].size - 1) for fn in filters} for ch in channels}
    used, missing = [], []
    for cyc in cycles:
        path = cfg["path_template"].format(root=cfg.get("root", ""), experiment=exp["name"],
                                           cycle=cyc, lead=cfg.get("lead", 0), **templates)
        if not os.path.isfile(path):
            if cfg.get("skip_missing", True):
                missing.append(cyc)
                continue
            raise FileNotFoundError(path)
        for ch, d in read_file(path, channels, cfg, need).items():
            x = get_var(d, xvar)
            e = edges[ch]
            inside = np.isfinite(x) & (x >= e[0]) & (x <= e[-1])
            xbin = np.clip(np.searchsorted(e, x, side="right") - 1, 0, e.size - 2)
            omb = d["obs"] - d["hofx"]
            for fn, fspec in filters.items():
                accumulate(acc[ch][fn], xbin, omb, inside & filter_mask(d, fspec.get("conditions")))
        used.append(cyc)
    print(f"[{exp['label']}] cycles used: {len(used)}, missing: {len(missing)}")
    if missing:
        print(f"[{exp['label']}]   missing: {' '.join(missing)}")
    return acc, edges, len(used)


def stat_values(acc, name, min_count):
    n = acc["n"]
    with np.errstate(divide="ignore", invalid="ignore"):
        v = STATS[name](n, acc["s"], acc["ss"])
    v[n < max(min_count, 1)] = np.nan
    return v


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def ramp(x, spec):
    """piecewise-linear error model (UFO ObsErrorModelRamp / Okamoto 2017 Eq. 4):
    err0 for x <= x0, err1 for x >= x1, linear in between"""
    x0, x1 = float(spec["x0"]), float(spec["x1"])
    e0, e1 = float(spec["err0"]), float(spec["err1"])
    return np.interp(x, [x0, x1], [e0, e1])


def plot(results, exps, channels, cfg, filters, ncyc):
    pcfg = cfg.get("plot", {}) or {}
    xvar = cfg["x"]["var"]
    xlabel = cfg["x"].get("label", VARS[xvar][1])
    stats = pcfg.get("stats") or [{"name": "std"}, {"name": "mean"}]
    curves = pcfg.get("curves") or [{"filter": fn} for fn in filters]
    min_count = int(pcfg.get("min_count", 10))
    show_bars = bool(pcfg.get("count_bars", True))
    ylim = pcfg.get("ylim")
    em = pcfg.get("error_model") or {}
    nrows, ncols = len(exps), len(channels)
    figsize = pcfg.get("figsize", [3.9 * ncols + 0.6, 3.4 * nrows + 1.6])
    default_colors = {"std": "tab:red", "mean": "tab:blue", "rms": "tab:green"}
    default_bars = ["0.85", "0.6", "0.4", "0.25"]
    lws = [1.0, 2.5, 1.8, 3.5]

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False,
                             constrained_layout=True)
    letters = "abcdefghijklmnopqrstuvwxyz"
    handles = {}
    for ie, exp in enumerate(exps):
        acc, edges = results[exp["label"]]
        for ic, ch in enumerate(channels):
            ax = axes[ie][ic]
            e = edges[ch]
            xc = 0.5 * (e[:-1] + e[1:])
            if show_bars:
                ax2 = ax.twinx()
                ax2.set_zorder(ax.get_zorder() - 1)   # bars behind the curves
                ax.patch.set_visible(False)
                for k, cv in enumerate(curves):
                    n = acc[ch][cv["filter"]]["n"]
                    with np.errstate(divide="ignore"):
                        ln = np.where(n > 0, np.log10(np.maximum(n, 1)), np.nan)
                    bar = ax2.bar(xc, ln, width=np.diff(e), align="center", lw=0,
                                  color=cv.get("bar_color", default_bars[k % 4]))
                    handles.setdefault(f"N: {filters[cv['filter']].get('label', cv['filter'])}",
                                       bar)
                ax2.set_ylim(*pcfg.get("count_ylim", [0, 7]))
                if ic == ncols - 1:
                    ax2.set_ylabel("log10(number)")
                else:
                    ax2.set_yticklabels([])
            for k, cv in enumerate(curves):
                fl = filters[cv["filter"]].get("label", cv["filter"])
                for st in stats:
                    name = st["name"]
                    if name not in STATS:
                        raise KeyError(f"unknown stat '{name}' (known: {', '.join(STATS)})")
                    y = stat_values(acc[ch][cv["filter"]], name, min_count)
                    (line,) = ax.plot(xc, y, color=st.get("color", default_colors.get(name, "k")),
                                      lw=cv.get("linewidth", lws[k % 4]),
                                      ls=cv.get("linestyle", "-"))
                    handles.setdefault(f"{st.get('label', name)}: {fl}", line)
            spec = per_channel(em.get("channels") if em.get("enabled") else None, ch)
            if spec:
                (line,) = ax.plot(xc, ramp(xc, spec), color="k", lw=1.5)
                handles.setdefault(em.get("label", "error model"), line)
            ax.axhline(0.0, color="0.5", lw=0.6, zorder=0)
            ax.set_xlim(e[0], e[-1])
            if ylim:
                ax.set_ylim(*per_channel(ylim, ch, ylim))
            ax.grid(ls=":", alpha=0.6)
            k = ie * ncols + ic
            ax.set_title(f"({letters[k % 26]})  {exp['label']}  Band{ch}", loc="left",
                         fontsize=10)
            ax.set_xlabel(xlabel if ie == nrows - 1 else "")
            if ic == 0:
                ax.set_ylabel(pcfg.get("ylabel", "SD and Mean of O−B (K)"))

    leg = dict(ncol=min(4, len(handles)), fontsize=8, frameon=False)
    try:                                   # matplotlib >= 3.7
        fig.legend(list(handles.values()), list(handles.keys()),
                   loc="outside lower center", **leg)
    except ValueError:                     # older matplotlib
        fig.legend(list(handles.values()), list(handles.keys()), loc="lower center",
                   bbox_to_anchor=(0.5, -0.06), **leg)
    bw = per_channel(cfg["x"].get("bin_width"), channels[0], 1.0)
    fig.suptitle(f"AHI Himawari-9 O−B statistics vs {xvar}  ({ncyc} cycles, lead "
                 f"{cfg.get('lead', 0)}h)\n{bias_text(cfg)[0]};  {xvar} bins {bw:g} K;  "
                 f"stats shown where N ≥ {min_count}", fontsize=10.5)
    return fig


def write_table(path, results, exps, channels, cfg, filters):
    stat_names = list(STATS)
    xvar = cfg["x"]["var"]
    with open(path, "w") as f:
        f.write(f"# O-B statistics binned by {xvar}; {bias_text(cfg)[0]}\n")
        for fn, fs in filters.items():
            f.write(f"# filter {fn}: {fs.get('label', fn)}  conditions={fs.get('conditions')}\n")
        f.write(f"{'experiment':<16s}{'ch':>4s} {'filter':<12s}{'x_lo':>8s}{'x_hi':>8s}"
                f"{'N':>12s}" + "".join(f"{s:>10s}" for s in stat_names) + "\n")
        for exp in exps:
            acc, edges = results[exp["label"]]
            for ch in channels:
                e = edges[ch]
                for fn in filters:
                    a = acc[ch][fn]
                    vals = {s: stat_values(a, s, 1) for s in stat_names}
                    for j in range(e.size - 1):
                        if a["n"][j] == 0:
                            continue
                        f.write(f"{exp['label']:<16s}{ch:>4d} {fn:<12s}{e[j]:>8.2f}"
                                f"{e[j + 1]:>8.2f}{int(a['n'][j]):>12d}"
                                + "".join(f"{vals[s][j]:>10.4f}" for s in stat_names) + "\n")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    with open(sys.argv[1]) as f:
        cfg = yaml.safe_load(f)

    channels = [int(c) for c in cfg.get("channels", [8, 9, 10, 13])]
    filters = cfg.get("filters") or {"all": {"label": "all valid obs", "conditions": []}}
    for cv in (cfg.get("plot", {}) or {}).get("curves") or []:
        if cv["filter"] not in filters:
            raise KeyError(f"curve filter '{cv['filter']}' not defined under filters")
    need = needed_inputs(cfg)
    cycles = build_cycles(cfg["cycles"])
    exps = normalize_experiments(cfg)
    btxt, btag = bias_text(cfg)
    print(f"cycles requested: {len(cycles)} ({cycles[0]} -> {cycles[-1]})\n{btxt}")

    results, ncyc = {}, 0
    for exp in exps:
        acc, edges, nused = collect(cfg, exp, cycles, channels, filters, need)
        results[exp["label"]] = (acc, edges)
        ncyc = max(ncyc, nused)

    ocfg = cfg.get("output", {}) or {}
    odir = ocfg.get("dir", "./figs")
    os.makedirs(odir, exist_ok=True)
    base = os.path.join(odir, f"{ocfg.get('prefix', 'ahi')}_OmB-vs-{cfg['x']['var']}_{btag}_"
                              f"{cycles[0]}-{cycles[-1]}_{cfg.get('lead', 0)}hr")
    fig = plot(results, exps, channels, cfg, filters, ncyc)
    out = f"{base}.{ocfg.get('format', 'png')}"
    fig.savefig(out, dpi=int(ocfg.get("dpi", 150)))
    plt.close(fig)
    print(f"saved {out}")
    if ocfg.get("table", True):
        write_table(f"{base}.txt", results, exps, channels, cfg, filters)
        print(f"saved {base}.txt")


if __name__ == "__main__":
    main()
