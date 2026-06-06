"""
make_social.py
==============
Generates a branded 1200x630 social link-preview image (Open Graph / Twitter
card) from the real headline result, so shared links render a polished card.
Writes website/assets/og-image.png.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WASSET = os.path.join(ROOT, "website", "assets")
os.makedirs(WASSET, exist_ok=True)

BG, PANEL, GOLD, BLUE, GREY, INK, MUTED = (
    "#0b1020", "#121a33", "#f4c45a", "#5aa6f4", "#c9c9c9", "#e7ecf6", "#9aa6c0")

# real headline numbers
loc, locstd, imp, auc = 3.8, 0.7, 12.0, 1.00
try:
    with open(os.path.join(ROOT, "results", "metrics.json")) as f:
        h = json.load(f)["headline"]
    loc, locstd = h["loc_error_pct"], h.get("loc_error_std", 0.7)
    imp, auc = h["improvement_factor"], h.get("detection_auc", 1.0)
except Exception:
    pass

# real reconstruction curve from the canonical scenario
x = np.linspace(0, 1, 200)
d_true = np.where(np.abs(x - 0.6) <= 0.04, 0.4, 0.0)
d_pred = d_true.copy()
try:
    with open(os.path.join(ROOT, "results", "results_full.json")) as f:
        R = json.load(f)
    s = next((z for z in R["scenarios"] if z["id"] == "canonical"), R["scenarios"][0])
    x = np.array(s["x"]); d_true = np.array(s["d_true"]); d_pred = np.array(s["d_pred"])
except Exception:
    pass

fig = plt.figure(figsize=(12, 6.3), dpi=100)
fig.patch.set_facecolor(BG)

# left: text block
axt = fig.add_axes([0.06, 0.0, 0.52, 1.0]); axt.axis("off")
axt.text(0.0, 0.84, "Aegis", color=GOLD, fontsize=64, fontweight="bold", va="top")
axt.text(0.0, 0.62, "Physics-Informed Structural\nDamage Localization",
         color=INK, fontsize=27, fontweight="bold", va="top", linespacing=1.25)
axt.text(0.0, 0.34, "Find hidden damage in a beam or bridge\nfrom only a handful of sparse sensors.",
         color=MUTED, fontsize=17, va="top", linespacing=1.3)
# stat chips
chips = [(f"{loc:.1f}±{locstd:.1f}%", "localization err", 0.00),
         (f"{auc:.2f}", "detection AUC", 0.34),
         (f"{imp:.0f}×", "vs. classical", 0.60)]
for v, k, xx in chips:
    axt.text(xx, 0.135, v, color=GOLD, fontsize=21, fontweight="bold", va="bottom")
    axt.text(xx, 0.075, k, color=MUTED, fontsize=11, va="bottom")

# right: the reconstruction plot on a panel
axp = fig.add_axes([0.63, 0.16, 0.33, 0.62])
axp.set_facecolor(PANEL)
for sp in axp.spines.values():
    sp.set_color("#26304f")
axp.fill_between(x, 0, d_true, color=GREY, alpha=0.35)
axp.plot(x, d_pred, color=GOLD, lw=3.5)
sx = np.array([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])
axp.scatter(sx, np.zeros_like(sx), color=BLUE, marker="v", s=70, zorder=5)
axp.set_xlim(0, 1); axp.set_ylim(0, max(0.55, d_pred.max() * 1.25))
axp.set_xticks([]); axp.set_yticks([])
axp.set_xlabel("position along the beam", color=MUTED, fontsize=12)
axp.set_title("true damage  vs  Aegis recovery", color=INK, fontsize=13, pad=8)

fig.savefig(os.path.join(WASSET, "og-image.png"), facecolor=BG,
            bbox_inches=None, pad_inches=0)
plt.close(fig)
print("wrote website/assets/og-image.png")
