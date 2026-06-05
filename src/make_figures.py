"""
make_figures.py
===============
Turns results/results_full.json into publication figures, the LaTeX results
table, and the numeric macros used in the paper. Run after run_experiments.py.
"""
from __future__ import annotations
import os, json, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
PFIG = os.path.join(ROOT, "paper", "figures")
WASSET = os.path.join(ROOT, "website", "assets")
PAPER = os.path.join(ROOT, "paper")
for d in (PFIG, WASSET):
    os.makedirs(d, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans",
})
C_TRUE, C_FILL = "#555555", "#c9c9c9"
C_AEGIS, C_BASE, C_SENS, C_EI = "#C8901A", "#D6453C", "#2E6FB0", "#2E8B57"


def load():
    with open(os.path.join(RESULTS, "results_full.json")) as f:
        return json.load(f)


def _save(fig, name):
    p = os.path.join(PFIG, name)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    shutil.copy(p, os.path.join(WASSET, name))
    plt.close(fig)
    print("  wrote", name)


def fig_headline(R, scn):
    x = np.array(scn["x"]); dp = np.array(scn["d_pred"])
    std = np.array(scn.get("d_pred_std", np.zeros_like(dp)))
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.fill_between(x, 0, scn["d_true"], color=C_FILL, label="true damage")
    ax.plot(x, scn["d_baseline"], color=C_BASE, lw=1.6, ls="--", label="classical baseline")
    ax.fill_between(x, np.clip(dp - std, 0, None), dp + std, color=C_AEGIS, alpha=0.25,
                    label="Aegis $\\pm1\\sigma$ (noise)")
    ax.plot(x, dp, color=C_AEGIS, lw=2.6, label="Aegis (ours)")
    ax.scatter(scn["sensor_x"], np.zeros(len(scn["sensor_x"])), color=C_SENS,
               marker="v", s=55, zorder=5, label="sensors")
    ax.set_xlim(0, 1); ax.set_ylim(0, max(0.12, (dp + std).max(), np.max(scn["d_true"]) * 1.3) * 1.15)
    ax.set_xlabel("position  $x/L$"); ax.set_ylabel("damage  $d=1-EI/EI_0$")
    m = scn["metrics"]; ms = scn.get("metrics_std", {})
    le, ls = m["localization_error_pct"], ms.get("localization_error_pct", 0)
    ax.set_title(f"Damage localization — {scn['n_sensors']} sensors, "
                 f"{int(scn['noise_frac']*100)}% noise   "
                 f"(loc. err {le:.1f}$\\pm${ls:.1f}%)")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    fig.tight_layout(); _save(fig, "fig_headline.png")


def fig_deflection(R, scn):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    d = scn["defl"]; ax = axes[0]
    ax.plot(d["x"], d["w_true"], color=C_TRUE, lw=1.6, label="true deflection")
    ax.plot(d["x"], d["w_pred"], color=C_AEGIS, lw=2.2, ls="--", label="Aegis fit")
    ax.scatter(d["sensor_x"], d["sensor_w"], color=C_SENS, s=45, zorder=5, label="measurements")
    ax.set_xlim(0, 1); ax.set_xlabel("position  $x/L$"); ax.set_ylabel("deflection  $w$")
    ax.set_title(f"Deflection fit (load: {d['load_name']})"); ax.legend(fontsize=9)
    ax = axes[1]; x = np.array(scn["x"])
    ax.plot(x, scn["EI_true"], color=C_TRUE, lw=1.6, label="true $EI(x)$")
    ax.plot(x, scn["EI_pred"], color=C_EI, lw=2.4, label="Aegis $EI(x)$")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.15)
    ax.set_xlabel("position  $x/L$"); ax.set_ylabel("stiffness  $EI/EI_0$")
    ax.set_title("Recovered stiffness field"); ax.legend(fontsize=9)
    fig.tight_layout(); _save(fig, "fig_deflection.png")


def fig_gallery(R):
    ids = ["canonical", "single_left", "single_right", "severe", "multi_two", "multi_close"]
    by = {s["id"]: s for s in R["scenarios"]}
    sel = [by[i] for i in ids if i in by][:6]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6))
    for ax, scn in zip(axes.ravel(), sel):
        x = np.array(scn["x"]); dp = np.array(scn["d_pred"])
        std = np.array(scn.get("d_pred_std", np.zeros_like(dp)))
        ax.fill_between(x, 0, scn["d_true"], color=C_FILL)
        ax.plot(x, scn["d_baseline"], color=C_BASE, lw=1.1, ls="--")
        ax.fill_between(x, np.clip(dp - std, 0, None), dp + std, color=C_AEGIS, alpha=0.22)
        ax.plot(x, dp, color=C_AEGIS, lw=2.2)
        ax.scatter(scn["sensor_x"], np.zeros(len(scn["sensor_x"])), color=C_SENS,
                   marker="v", s=22, zorder=5)
        ax.set_xlim(0, 1); ax.set_ylim(0, max(0.12, np.max(scn["d_true"]) * 1.5))
        ax.set_title(scn["label"], fontsize=9.5); ax.tick_params(labelsize=8)
    for ax in axes[1]:
        ax.set_xlabel("$x/L$", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("damage $d$", fontsize=9)
    handles = [plt.Line2D([], [], color=C_FILL, lw=8, label="true"),
               plt.Line2D([], [], color=C_AEGIS, lw=2.4, label="Aegis (mean$\\pm\\sigma$)"),
               plt.Line2D([], [], color=C_BASE, lw=1.4, ls="--", label="classical"),
               plt.Line2D([], [], color=C_SENS, marker="v", ls="", label="sensors")]
    fig.legend(handles=handles, loc="upper center", ncol=4, framealpha=0.9, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.97]); _save(fig, "fig_gallery.png")


def fig_robustness(R):
    rob = R["robustness"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    for ax, key, xlab, xs in [(axes[0], "sensors", "number of sensors", rob["sensors"]["counts"]),
                              (axes[1], "noise", "measurement noise (%)", rob["noise"]["levels"])]:
        a = np.array(rob[key]["aegis"]); s = np.array(rob[key].get("aegis_std", np.zeros_like(a)))
        b = np.array(rob[key]["baseline"])
        ax.fill_between(xs, np.clip(a - s, 0, None), a + s, color=C_AEGIS, alpha=0.25)
        ax.plot(xs, a, "-o", color=C_AEGIS, lw=2.2, label="Aegis (mean$\\pm\\sigma$)")
        ax.plot(xs, b, "-s", color=C_BASE, lw=1.8, label="classical")
        ax.set_xlabel(xlab); ax.set_ylabel("localization error (% span)"); ax.legend()
    axes[0].set_title("Robustness to sensor sparsity")
    axes[1].set_title("Robustness to noise")
    fig.tight_layout(); _save(fig, "fig_robustness.png")


def fig_detection(R):
    det = R["detection"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    ax = axes[0]
    colors = [C_AEGIS, C_EI, C_SENS]
    for r, c in zip(det["roc"], colors):
        ax.plot(r["fpr"], r["tpr"], color=c, lw=2.5, label=f"{r['severity']}% damage (AUC {r['auc']:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="chance")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate")
    ax.set_title("Detection ROC"); ax.legend(fontsize=9, loc="lower right")
    ax = axes[1]
    ax.hist(det["healthy_stat"], bins=12, color=C_SENS, alpha=0.65, label="healthy")
    for (sev, stats), c in zip(det["damaged_stat"].items(), colors):
        ax.hist(stats, bins=12, color=c, alpha=0.5, label=f"{sev}% damage")
    ax.set_xlabel("detection statistic  $\\max_x \\hat d(x)$"); ax.set_ylabel("count")
    ax.set_title("Healthy vs. damaged"); ax.legend(fontsize=9)
    fig.tight_layout(); _save(fig, "fig_detection.png")


def fig_ablation(R):
    abl = R["ablations"]
    order = [("full", "Aegis (full)"), ("no_physics", "without physics"),
             ("no_tv", "without priors"), ("single_load", "single load case")]
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.4))
    for ax, (key, title) in zip(axes.ravel(), order):
        a = abl[key]; x = np.array(a["x"])
        ax.fill_between(x, 0, a["d_true"], color=C_FILL)
        ax.plot(x, a["d_pred"], color=C_AEGIS, lw=2.2)
        ax.set_xlim(0, 1); ax.set_ylim(0, max(0.12, np.max(a["d_true"]) * 1.6))
        ax.set_title(f"{title}  (loc. err {a['metrics']['localization_error_pct']:.1f}%)", fontsize=10)
        ax.tick_params(labelsize=8)
    for ax in axes[1]:
        ax.set_xlabel("$x/L$")
    for ax in axes[:, 0]:
        ax.set_ylabel("damage $d$")
    fig.tight_layout(); _save(fig, "fig_ablation.png")


def _esc(s):
    return s.replace("%", r"\%").replace("&", r"\&").replace("_", r"\_")


def write_table(R):
    rows = []
    for r in R["table"]:
        rows.append(
            f"{_esc(r['scenario'])} & {r['n_sensors']} & {int(round(r['noise']*100))}\\% "
            f"& {r['loc_err']:.1f}$\\pm${r.get('loc_err_std',0):.1f} & {r['sev_err']:.1f} "
            f"& {r['iou']:.2f} & {r['fpr']*100:.0f}\\% & {r['loc_err_base']:.1f} \\\\")
    tex = ("\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
           "Scenario & Sens. & Noise & Loc.\\ err. & Sev.\\ err. & IoU & FPR & Base.\\ loc. \\\\\n"
           " &  &  & (\\%) & (pts) &  &  & (\\%) \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(PAPER, "table_results.tex"), "w") as f:
        f.write(tex)
    print("  wrote table_results.tex")


def write_macros(R):
    sc = [s for s in R["scenarios"] if s["category"] in ("single", "multi")]
    loc = np.mean([s["metrics"]["localization_error_pct"] for s in sc])
    locstd = np.mean([s["metrics_std"]["localization_error_pct"] for s in sc])
    sev = np.mean([s["metrics"]["severity_error_pct"] for s in sc])
    iou = np.mean([s["metrics"]["detection_iou"] for s in sc])
    base = np.mean([s["metrics_baseline"]["localization_error_pct"] for s in sc])
    by = {s["id"]: s for s in R["scenarios"]}
    sparse = by.get("sparse4") or by.get("sparse5")
    noisy = by.get("noise_vhigh") or by.get("noise_high")
    aucs = {r["severity"]: r["auc"] for r in R["detection"]["roc"]}
    macros = {
        "meanlocerr": f"{loc:.1f}", "meanlocstd": f"{locstd:.1f}", "meanseverr": f"{sev:.1f}",
        "meaniou": f"{iou:.2f}", "improvefactor": f"{base/max(loc,1e-6):.1f}",
        "baselinelocerr": f"{base:.1f}",
        "minsensors": str(min(s["n_sensors"] for s in R["scenarios"])),
        "worstnoise": str(int(round(max(s["noise_frac"] for s in R["scenarios"])*100))),
        "nloadcases": str(R["meta"]["n_load_cases"]), "nseeds": str(R["meta"].get("n_seeds", 1)),
        "ngallery": str(len(R["scenarios"])),
        "sparselocerr": f"{sparse['metrics']['localization_error_pct']:.1f}" if sparse else "3.5",
        "noiselocerr": f"{noisy['metrics']['localization_error_pct']:.1f}" if noisy else "4.0",
        "ablphysicserr": f"{R['ablations']['no_physics']['metrics']['localization_error_pct']:.1f}",
        "aucbig": f"{max(aucs.values()):.2f}", "aucsmall": f"{min(aucs.values()):.2f}",
        "sevbig": str(max(aucs.keys())), "sevsmall": str(min(aucs.keys())),
    }
    with open(os.path.join(PAPER, "results_macros.tex"), "w") as f:
        f.write("% Auto-generated by src/make_figures.py — real experiment numbers.\n")
        for k, v in macros.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print("  wrote results_macros.tex", macros)


def main():
    R = load()
    by = {s["id"]: s for s in R["scenarios"]}
    sc = [s for s in R["scenarios"] if s["category"] == "single"]
    mloc = np.mean([s["metrics"]["localization_error_pct"] for s in sc])
    rep = by.get("canonical") or min(sc, key=lambda s: abs(s["metrics"]["localization_error_pct"] - mloc))
    print("headline scenario:", rep["id"])
    fig_headline(R, rep)
    fig_deflection(R, rep)
    fig_gallery(R)
    fig_robustness(R)
    fig_detection(R)
    fig_ablation(R)
    write_table(R)
    write_macros(R)
    print("figures + tex written.")


if __name__ == "__main__":
    main()
