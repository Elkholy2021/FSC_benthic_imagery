import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib import cm

GWR = LinearSegmentedColormap.from_list(
    "grn_wht_red",
    ["#7a0d0d", "#c0392b", "#e8847a", "#f7d6d0", "#ffffff",
     "#d4ead7", "#8fc99a", "#3f9d5b", "#0f6b2f"])


Ta = 2.0          # seconds per annotation point
k  = 0.25         # review-time factor
Tr = k * Ta       # review time per point 
MS_F = 100        # manually labelled set, FSC
MS_T = 1100       # manually labelled set, TFM

TpF = 10   # ML processing time, FSC, average over the three datasets
TpT = 100  # ML processing time, TFM, average over the three datasets




datasets = {
    "18167": dict(S=3588, accF=0.577, stdF=0.023, accT=0.756, stdT=0.007,
                  color="#6a2d8b"),
    "18290": dict(S=3613, accF=0.739, stdF=0.023, accT=0.839, stdT=0.005,
                  color="#0d6e8c"),
    "16817": dict(S=9891, accF=0.701, stdF=0.016, accT=0.794, stdT=0.02,
                  color="#b35c00"),
}   #filled manually with the mean and std of the accuracy for each dataset, for both FSC and TFM


S_grid    = np.linspace(400, 15000, 500)
dacc_grid = np.linspace(0.02, 0.35, 500)
SS, DD = np.meshgrid(S_grid, dacc_grid)


def pipeline_times_min(S, accF, accT):
    """T_MANUAL, T_FSC, T_TFM in minutes."""
    print(f"S={S}, accF={accF:.3f}, accT={accT:.3f}")
    T_man = S * (Ta + Tr)
    T_fsc = MS_F * Ta + TpF + Tr * S + Ta * (1 - accF) * (S - MS_F)
    T_tfm = MS_T * Ta + TpT + Tr * S + Ta * (1 - accT) * (S - MS_T)
    return T_man / 60.0, T_fsc / 60.0, T_tfm / 60.0


def time_saved_minutes_accT(S, dacc, accT):
    """Background field: acc_FSC = accT - dacc (fixed acc_TFM anchor)."""
    accF = accT - dacc
    base = (MS_T - MS_F) * Ta + (TpT - TpF)
    corr = Ta * ((1 - accT) * (S - MS_T) - (1 - accF) * (S - MS_F))
    return (base + corr) / 60.0


def saved_halfrange_min(S, stdF, stdT):
    """Best/worst-case half-range of T_TFM - T_FSC (independent of k)."""
    return Ta * (stdT * (S - MS_T) + stdF * (S - MS_F)) / 60.0


# Shared background anchored on acc_TFM = 0.80.
accT_bg = 0.8
Z = time_saved_minutes_accT(SS, DD, accT_bg)
vmax = max(np.abs(Z).max(), 1)
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
levels = np.linspace(-vmax, vmax, 25)

fig, ax = plt.subplots(figsize=(11.5, 8.0))
fig.suptitle(
    r"Time saved by FSC over TFM:  $T_{TFM}-T_{FSC}$  (minutes)",
    fontsize=14, y=0.975)
ax.set_title(
    rf"$T_a=2$ s,  $k={k}$,  $MS_{{FSC}}=100$,  $MS_{{TFM}}=1100$,$T_p^{{FSC}}=10$ s,  $T_p^{{TFM}}=100$ s  "
    rf"(background at acc$_{{TFM}}$={accT_bg*100:.0f})",
    
    fontsize=11)

# Display the accuracy-gap (y) dimension in percentage points (0-100 scale),
# while all time/physics computations stay in fraction units.
DDp = DD * 100.0

cf = ax.contourf(SS, DDp, Z, levels=levels, cmap=GWR, norm=norm, extend="both")

be = ax.contour(SS, DDp, Z, levels=[0], colors="#0d3b66", linewidths=2.6)
ax.clabel(be, fmt={0: "break-even"}, fontsize=10, inline=True)

other = [l for l in (-200, -150, -100, -60, -10, 10, 60, 100, 150, 200)
         if -vmax < l < vmax]
if other:
    cs = ax.contour(SS, DDp, Z, levels=sorted(other),
                    colors="#555555", linewidths=0.6, linestyles="-")
    ax.clabel(cs, fmt=lambda v: f"{v:.0f} min", fontsize=8, inline=True)

box_pos = {"18167": (0.515, 0.86), "18290": (0.555, 0.52),
           "16817": (0.40, 0.12)} #formatting of text boxes

for name, p in datasets.items():
    S, accF, stdF, accT, stdT = p["S"], p["accF"], p["stdF"], p["accT"], p["stdT"]
    col = p["color"]
    dacc = accT - accF
    half = stdF + stdT                          # gap half-range (whisker)

    T_man, T_fsc, T_tfm = pipeline_times_min(S, accF, accT)
    saved = T_tfm - T_fsc                        # + => FSC saves
    hr = saved_halfrange_min(S, stdF, stdT)      # +/- on the savings

    _, T_fsc_hiacc, _ = pipeline_times_min(S, accF + stdF, accT)
    _, T_fsc_loacc, _ = pipeline_times_min(S, accF - stdF, accT)
    hf = (T_fsc_loacc - T_fsc_hiacc) / 2.0       # +/- on T_FSC
    _, _, T_tfm_hiacc = pipeline_times_min(S, accF, accT + stdT)
    _, _, T_tfm_loacc = pipeline_times_min(S, accF, accT - stdT)
    ht = (T_tfm_loacc - T_tfm_hiacc) / 2.0       # +/- on T_TFM

    # whisker = best/worst-case scenario on the accuracy gap (in pp)
    ax.errorbar(S, dacc * 100, yerr=half * 100, fmt="none",
                ecolor=col, elinewidth=3.0, capsize=7, capthick=2.2,
                zorder=6)
    ax.plot(S, dacc * 100, "o", ms=11, mfc="white", mec=col, mew=2.2, zorder=7)

    if saved >= 0:                               # FSC wins
        who, loser = "FSC", "TFM"
        sv, pct = saved, 100.0 * saved / T_tfm
        man_save = T_man - T_fsc
    else:                                        # TFM wins
        who, loser = "TFM", "FSC"
        sv, pct = -saved, 100.0 * (-saved) / T_fsc
        man_save = T_man - T_tfm
    man_pct = 100.0 * man_save / T_man

    # bold "WHO saves X +/- Y min (Z%)" via mathtext (DejaVu Sans bold matches)
    bold = (r"$\mathbf{" + who + r"\ saves\ " + f"{sv:.1f}" + r"\pm" +
            f"{hr:.1f}" + r"\ min\ (" + f"{pct:.1f}" + r"\%)}$")
    line_saved = bold + f" compared to {loser}"
    dataset_dict = {"18167": "A", "18290": "B", "16817": "C"}
    dataset = dataset_dict.get(name, name)
    txt = (
        f"Dataset {dataset_dict[name]} - {name}   (S={S:,})\n"
        f" T_MANUAL: {T_man:.1f}, T_FSC: {T_fsc:.1f} $\\pm$ {hf:.1f}, "
        f"T_TFM: {T_tfm:.1f} $\\pm$ {ht:.1f} (min)\n"
        rf"acc$_F$={accF*100:.1f} $\pm$ {stdF*100:.1f},  "
        rf"acc$_T$={accT*100:.1f} $\pm$ {stdT*100:.1f},  "
        rf"$\Delta$acc={dacc*100:.1f} $\pm$ {half*100:.1f}"
        f"\n{line_saved}\n"
        f"And {man_save:.1f} min ({man_pct:.1f}%) compared to manual"
    )

    ax.annotate(
        txt, xy=(S, dacc * 100), xytext=box_pos[name],
        textcoords="axes fraction",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=col,
                  lw=1.4, alpha=0.95),
        arrowprops=dict(arrowstyle="-", color=col, lw=1.2))

ax.set_xlabel("Dataset size  $S$", fontsize=12)
ax.set_ylabel(r"Accuracy gap  $\Delta$acc $=$ acc$_{TFM}-$acc$_{FSC}$  (%)",
              fontsize=12)
ax.set_xlim(S_grid.min(), S_grid.max())
ax.set_ylim(dacc_grid.min() * 100, dacc_grid.max() * 100)
ax.grid(alpha=0.15)

from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase


class HandlerWhisker(HandlerBase):
    """Legend swatch shaped like a capped error bar:  |-----|"""
    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                        width, height, fontsize, trans):
        ymid = height / 2.0 - ydescent
        x0, x1 = -xdescent, width - xdescent
        col = orig_handle.get_color()
        lw = orig_handle.get_linewidth()
        cap = height * 0.75                      # cap height
        bar = Line2D([x0, x1], [ymid, ymid], color=col, lw=lw,
                     solid_capstyle="butt")
        capL = Line2D([x0, x0], [ymid - cap / 2, ymid + cap / 2],
                      color=col, lw=max(lw - 0.8, 1.2))
        capR = Line2D([x1, x1], [ymid - cap / 2, ymid + cap / 2],
                      color=col, lw=max(lw - 0.8, 1.2))
        for a in (bar, capL, capR):
            a.set_transform(trans)
        return [bar, capL, capR]


whisker_proxy = Line2D([0], [0], color="0.2", lw=3.0,
                       label="$\\Delta$acc best/worst case")
mean_proxy = Line2D([0], [0], marker="o", linestyle="none",
                    markerfacecolor="white", markeredgecolor="0.2",
                    markeredgewidth=2.2, markersize=9,
                    label="Mean $\\Delta$acc")
legend_elems = [
    Line2D([0], [0], color="#0d3b66", lw=2.6, label="Break-even (0 min)"),
    Line2D([0], [0], color="#555555", lw=0.6, label="Other isolines"),
    mean_proxy,
    whisker_proxy,
]
ax.legend(handles=legend_elems, loc="upper left", fontsize=9, framealpha=0.95,
          handler_map={whisker_proxy: HandlerWhisker()})

cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=GWR), ax=ax,
                    fraction=0.046, pad=0.02,
                    ticks=np.linspace(-vmax, vmax, 9))
cbar.set_label("FSC saves (+)   /   TFM saves (\u2212)   [min]", fontsize=11)

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("time_saved_fsc_vs_tfm_std.png", dpi=160, bbox_inches="tight")
print("saved  (background acc_TFM = %.2f, k = %.2f)" % (accT_bg, k))
for name, p in datasets.items():
    S, accF, stdF, accT, stdT = p["S"], p["accF"], p["stdF"], p["accT"], p["stdT"]
    T_man, T_fsc, T_tfm = pipeline_times_min(S, accF, accT)
    saved = T_tfm - T_fsc
    hr = saved_halfrange_min(S, stdF, stdT)
    hf = Ta * stdF * (S - MS_F) / 60.0
    ht = Ta * stdT * (S - MS_T) / 60.0
    who = "FSC" if saved >= 0 else "TFM"
    print(f"{name}: T_MAN={T_man:.1f} T_FSC={T_fsc:.1f}+/-{hf:.1f} "
          f"T_TFM={T_tfm:.1f}+/-{ht:.1f}  "
          f"{who} saves {abs(saved):.1f} +/- {hr:.1f} min")
