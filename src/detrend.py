"""Photobleaching correction: fit a decay model to each wavelength and subtract it.

Usage: python src/detrend.py [data_dir] [out_dir]
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "/Users/ivan/Downloads/2026_03_10-16_56_33")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent / "output")
CONTEXT_S = 10.0
MASK_S = 15.0  # exclude +-MASK_S around events from the fit so the response doesn't bias it
WAVES = {"CH1-410": "410 нм", "CH1-470": "470 нм", "CH1-560": "560 нм"}
COLORS = {"CH1-410": "#6f42c1", "CH1-470": "#1f77b4", "CH1-560": "#d62728"}

OUT.mkdir(parents=True, exist_ok=True)
fl = pd.read_csv(DATA / "Fluorescence.csv", skiprows=1, index_col=False)[["TimeStamp", *WAVES]]
fl["t"] = fl["TimeStamp"] / 1000
ev = pd.read_csv(DATA / "Events.csv")
ev["t"] = ev["TimeStamp"] / 1000
windows = {n: (g["t"].min(), g["t"].max()) for n, g in ev.groupby("Name")}

mask = np.ones(len(fl), bool)
for t0, t1 in windows.values():
    mask &= ~((fl["t"] > t0 - MASK_S) & (fl["t"] < t1 + MASK_S))

def exp1(t, a, k, c): return a * np.exp(-t / k) + c
def exp2(t, a1, k1, a2, k2, c): return a1 * np.exp(-t / k1) + a2 * np.exp(-t / k2) + c
def lin(t, a, b): return a * t + b

t = fl["t"].to_numpy()
models = {
    "линейная": (lin, lambda y: [(y[-1] - y[0]) / t[-1], y[0]]),
    "1 экспонента": (exp1, lambda y: [y[0] - y[-1], t[-1], y[-1]]),
    "2 экспоненты": (exp2, lambda y: [(y[0] - y[-1]) / 2, 50, (y[0] - y[-1]) / 2, t[-1], y[-1]]),
}
fits, rows = {}, []
for w in WAVES:
    y = fl[w].to_numpy()
    for mname, (f, p0) in models.items():
        try:
            p, _ = curve_fit(f, t[mask], y[mask], p0=p0(y), maxfev=50000)
            pred = f(t, *p)
            res = (y - pred)[mask]
            fits[(w, mname)] = pred
            rows.append({"wavelength": w, "model": mname, "rmse": np.sqrt((res ** 2).mean()),
                         "params": np.round(p, 3).tolist()})
        except RuntimeError:
            rows.append({"wavelength": w, "model": mname, "rmse": np.nan, "params": "no convergence"})
cmp = pd.DataFrame(rows)
cmp.to_csv(OUT / "fit_comparison.csv", index=False)
print(cmp.to_string(index=False))

# single exponential: 2-exp gains little and is degenerate for 560 nm (+-600 cancelling terms)
best = {w: "1 экспонента" for w in WAVES}

for w in WAVES:
    fit = fits[(w, best[w])]
    fl[w + "_fit"] = fit
    fl[w + "_sub"] = fl[w] - fit               # absolute residual
    fl[w + "_dff"] = (fl[w] - fit) / fit * 100  # % of the decay curve
fl.drop(columns="TimeStamp").to_csv(OUT / "detrended.csv", index=False)

# 1) fit overview
fig, axes = plt.subplots(len(WAVES), 2, figsize=(15, 8), sharex=True)
for i, w in enumerate(WAVES):
    ax = axes[i, 0]
    ax.plot(t, fl[w], color=COLORS[w], lw=0.5, label="данные")
    for mname, ls in zip(models, [":", "--", "-"]):
        if (w, mname) in fits:
            ax.plot(t, fits[(w, mname)], "k", ls=ls, lw=1, label=mname)
    ax.set_ylabel(WAVES[w]); ax.grid(alpha=0.3)
    ax = axes[i, 1]
    ax.plot(t, fl[w + "_dff"], color=COLORS[w], lw=0.5)
    for n, (t0, t1) in windows.items():
        ax.axvspan(t0 - 0.5, t1 + 0.5, color="orange", alpha=0.7)
    ax.axhline(0, color="k", lw=0.5); ax.set_ylabel(f"{WAVES[w]}: данные − спад, %"); ax.grid(alpha=0.3)
axes[0, 0].legend(fontsize=8, ncol=4)
axes[-1, 0].set_xlabel("время, с"); axes[-1, 1].set_xlabel("время, с")
fig.tight_layout(); fig.savefig(OUT / "detrend_overview.png", dpi=150); plt.close(fig)

# 2) events after correction
fig, axes = plt.subplots(len(WAVES), len(windows), figsize=(15, 8), sharex=True)
for j, (n, (t0, t1)) in enumerate(sorted(windows.items())):
    seg = fl[(fl["t"] >= t0 - CONTEXT_S) & (fl["t"] <= t1 + CONTEXT_S)]
    for i, w in enumerate(WAVES):
        ax = axes[i, j]
        ax.plot(seg["t"] - t0, seg[w + "_dff"], color=COLORS[w], lw=1, marker=".", ms=3)
        ax.axvspan(0, t1 - t0, color="orange", alpha=0.35)
        ax.axhline(0, color="k", lw=0.5); ax.grid(alpha=0.3)
        if i == 0: ax.set_title(f"{n} (t={t0:.1f} с)")
        if j == 0: ax.set_ylabel(f"{WAVES[w]}\nотклонение от спада, %")
        if i == len(WAVES) - 1: ax.set_xlabel("время от начала события, с")
fig.suptitle("События после вычитания экспоненциального спада")
fig.tight_layout(); fig.savefig(OUT / "events_detrended.png", dpi=150)
