"""Fast tuning harness: train on the canonical scenario and print metrics + an
ASCII view of the recovered damage field.  Usage:
   python tune.py [w_l1] [w_tv] [d_max] [adam]
"""
import sys, time, numpy as np
from beam_simulation import BeamModel
from datagen import build_dataset, DamageSegment
from pinn_model import AegisPINN, AegisConfig
from evaluate import all_metrics

w_l1  = float(sys.argv[1]) if len(sys.argv) > 1 else 3e-3
w_tv  = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3
d_max = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9
adam  = int(sys.argv[4])   if len(sys.argv) > 4 else 3000

beam = BeamModel(L=1.0, n_elem=200, EI0=1.0, bc="ss")
segs = [DamageSegment(center=0.6, half_width=0.04, severity=0.40)]
data = build_dataset(beam, segs, n_sensors=7, noise_frac=0.02, seed=1)

cfg = AegisConfig(n_grid=400, adam_iters=adam, lbfgs_iters=80, lr=5e-3,
                  w_l1=w_l1, w_tv=w_tv, d_max=d_max, seed=0)
pinn = AegisPINN(cfg, data["loads"])
pinn.set_data(data["sensor_x"], data["measurements"], data["w_scale"])

t0 = time.time()
pinn.train(verbose=True, log_every=adam // 3)
dt = time.time() - t0

x = data["x_dense"]
dp = pinn.predict_damage(x)
dt_true = data["d_true_dense"]
m = all_metrics(x, dp, dt_true)

def spark(vals, lo=0.0, hi=0.6):
    chars = " .:-=+*#%@"
    out = ""
    for v in np.interp(np.linspace(0, 1, 60), np.linspace(0, 1, len(vals)), vals):
        t = max(0, min(0.999, (v - lo) / (hi - lo)))
        out += chars[int(t * (len(chars) - 1))]
    return out

print(f"\ncfg: w_l1={w_l1} w_tv={w_tv} d_max={d_max} adam={adam}  ({dt:.0f}s)")
print(f"true  |{spark(dt_true)}|  peak={np.max(dt_true):.2f}@{x[np.argmax(dt_true)]:.2f}")
print(f"pred  |{spark(dp)}|  peak={np.max(dp):.2f}@{x[np.argmax(dp)]:.2f}")
print(f"metrics: loc_err={m['localization_error_pct']:.1f}%  sev_err={m['severity_error_pct']:.1f}pts  "
      f"rmse={m['damage_rmse']:.3f}  IoU={m['detection_iou']:.2f}  FPR={m['false_positive_rate']:.2f}")
