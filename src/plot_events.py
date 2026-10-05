"""Fluorescence (410/470/560) around stimulation events.

Usage: python src/plot_events.py [data_dir] [out_dir]
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "/Users/ivan/Downloads/2026_03_10-16_56_33")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent / "output")
CONTEXT_S = 10.0   # seconds shown on each side of the stimulation window
BASELINE_S = 5.0   # pre-event baseline length for normalisation
WAVES = {"CH1-410": "410 нм", "CH1-470": "470 нм", "CH1-560": "560 нм"}
COLORS = {"CH1-410": "#6f42c1", "CH1-470": "#1f77b4", "CH1-560": "#d62728"}

OUT.mkdir(parents=True, exist_ok=True)

# Fluorescence.csv: 1st line is a JSON config, header on the 2nd; timestamps in ms
fl = pd.read_csv(DATA / "Fluorescence.csv", skiprows=1, index_col=False)
fl = fl[["TimeStamp", *WAVES]]
fl["t"] = fl["TimeStamp"] / 1000
ev = pd.read_csv(DATA / "Events.csv")
ev["t"] = ev["TimeStamp"] / 1000

rows = []
windows = {}
for name, g in ev.groupby("Name"):
    # in the file each event has State 0 first and State 1 afterwards; the window is between them
    windows[name] = (g["t"].min(), g["t"].max())
    t0, t1 = windows[name]
    inside = fl[(fl["t"] >= t0) & (fl["t"] <= t1)]
    base = fl[(fl["t"] >= t0 - BASELINE_S) & (fl["t"] < t0)]
    for w in WAVES:
        rows.append({
            "event": name, "start_s": t0, "end_s": t1, "duration_s": t1 - t0,
            "wavelength": w, "n_samples": len(inside),
            "baseline_mean": base[w].mean(), "window_mean": inside[w].mean(),
            "change_pct": (inside[w].mean() / base[w].mean() - 1) * 100 if len(inside) else float("nan"),
        })
summary = pd.DataFrame(rows)
summary.to_csv(OUT / "event_summary.csv", index=False)
print(summary.round(3).to_string(index=False))

for mode in ("raw", "pct"):
    fig, axes = plt.subplots(len(WAVES), len(windows), figsize=(5 * len(windows), 8), sharex=True)
    for j, (name, (t0, t1)) in enumerate(sorted(windows.items())):
        seg = fl[(fl["t"] >= t0 - CONTEXT_S) & (fl["t"] <= t1 + CONTEXT_S)]
        base = fl[(fl["t"] >= t0 - BASELINE_S) & (fl["t"] < t0)]
        for i, w in enumerate(WAVES):
            ax = axes[i, j]
            y = seg[w] if mode == "raw" else (seg[w] / base[w].mean() - 1) * 100
            ax.plot(seg["t"] - t0, y, color=COLORS[w], lw=1, marker=".", ms=3)
            ax.axvspan(0, t1 - t0, color="orange", alpha=0.35, label="между 0 и 1")
            ax.axvline(0, color="k", lw=0.6)
            ax.axvline(t1 - t0, color="k", lw=0.6)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(f"{name}  (t={t0:.1f} с, окно {t1 - t0:.2f} с)")
            if j == 0:
                ax.set_ylabel(f"{WAVES[w]}\n" + ("флуоресценция, а.е." if mode == "raw" else "изменение, % от базовой"))
            if i == len(WAVES) - 1:
                ax.set_xlabel("время от начала события, с")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Флуоресценция по длинам волн вокруг стимуляции" + ("" if mode == "raw" else f" (норм. на {BASELINE_S:.0f} с до события)"))
    fig.tight_layout()
    fig.savefig(OUT / f"events_{mode}.png", dpi=150)
    plt.close(fig)

# overview of the whole recording with events
fig, axes = plt.subplots(len(WAVES), 1, figsize=(14, 7), sharex=True)
for ax, w in zip(axes, WAVES):
    ax.plot(fl["t"], fl[w], color=COLORS[w], lw=0.6)
    for name, (t0, t1) in windows.items():
        ax.axvspan(t0, t1, color="orange", alpha=0.6)
        ax.annotate(name, (t0, ax.get_ylim()[1]), fontsize=8, ha="center", va="top")
    ax.set_ylabel(WAVES[w])
    ax.grid(alpha=0.3)
axes[-1].set_xlabel("время, с")
fig.tight_layout()
fig.savefig(OUT / "overview.png", dpi=150)
