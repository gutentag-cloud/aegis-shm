"""
run_experiments.py
==================
Runs the full Aegis experiment suite with statistical rigor and writes:
  * results/results_full.json  -- complete arrays for figures
  * results/metrics.json       -- compact metrics summary
  * website/data/results.js    -- window.AEGIS_RESULTS = {...} for the site

Each scenario / robustness point is evaluated over several independent noise
realizations (Monte-Carlo), so metrics are reported as mean +/- std and the
recovered damage map carries an ensemble uncertainty band.  A separate
detection study (healthy vs. damaged trials) yields ROC curves and AUC.

Usage:
  python run_experiments.py            # full suite (~25 min)
  python run_experiments.py --quick    # fast smoke run
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np
import torch

from beam_simulation import BeamModel
from datagen import build_dataset, DamageSegment
from pinn_model import AegisPINN, AegisConfig
from baselines import curvature_baseline
from evaluate import all_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
WEBDATA = os.path.join(ROOT, "website", "data")
MODELS = os.path.join(ROOT, "models")
for d in (RESULTS, WEBDATA, MODELS):
    os.makedirs(d, exist_ok=True)


def seg(c, hw, s):
    return DamageSegment(center=c, half_width=hw, severity=s)


# (id, label, category, segments, n_sensors, noise_frac)
GALLERY = [
    ("canonical",    "Canonical: damage @ 0.60L (40%)",    "single",  [seg(0.60, 0.04, 0.40)], 7, 0.02),
    ("single_mid",   "Single damage @ 0.50L (35%)",        "single",  [seg(0.50, 0.04, 0.35)], 7, 0.02),
    ("single_left",  "Single damage @ 0.30L (40%)",        "single",  [seg(0.30, 0.04, 0.40)], 7, 0.02),
    ("single_right", "Single damage @ 0.70L (30%)",        "single",  [seg(0.70, 0.04, 0.30)], 7, 0.02),
    ("severe",       "Severe damage @ 0.60L (55%)",        "single",  [seg(0.60, 0.05, 0.55)], 7, 0.02),
    ("mild",         "Mild damage @ 0.45L (20%)",          "single",  [seg(0.45, 0.04, 0.20)], 9, 0.02),
    ("multi_two",    "Two damages @ 0.30L & 0.65L",        "multi",   [seg(0.30, 0.04, 0.35), seg(0.65, 0.04, 0.30)], 9, 0.02),
    ("multi_close",  "Two close damages @ 0.55L & 0.72L",  "multi",   [seg(0.55, 0.035, 0.35), seg(0.72, 0.035, 0.35)], 11, 0.02),
    ("noise_high",   "Single @ 0.60L, 5% noise",           "noise",   [seg(0.60, 0.04, 0.40)], 7, 0.05),
    ("noise_vhigh",  "Single @ 0.60L, 8% noise",           "noise",   [seg(0.60, 0.04, 0.40)], 7, 0.08),
    ("sparse5",      "Single @ 0.60L, only 5 sensors",     "sparsity",[seg(0.60, 0.04, 0.40)], 5, 0.02),
    ("sparse4",      "Single @ 0.40L, only 4 sensors",     "sparsity",[seg(0.40, 0.04, 0.40)], 4, 0.02),
    ("dense11",      "Single @ 0.60L, 11 sensors",         "sparsity",[seg(0.60, 0.04, 0.40)], 11, 0.02),
]


# --------------------------------------------------------------------------- #
#  One training run                                                           #
# --------------------------------------------------------------------------- #
def run_one(segs, n_sensors, noise_frac, cfg_kwargs, seed=1, bc="ss", save_to=None):
    beam = BeamModel(L=1.0, n_elem=200, EI0=1.0, bc=bc)
    data = build_dataset(beam, segs, n_sensors=n_sensors, noise_frac=noise_frac, seed=seed)
    cfg = AegisConfig(bc=bc, **cfg_kwargs)
    pinn = AegisPINN(cfg, data["loads"])
    pinn.set_data(data["sensor_x"], data["measurements"], data["w_scale"])
    pinn.train(verbose=False)
    if save_to is not None:
        torch.save({"state_dict": pinn.d_net.state_dict(), "cfg": cfg.__dict__}, save_to)

    x = data["x_dense"]
    d_pred = pinn.predict_damage(x)
    EI_pred = pinn.predict_EI(x)
    _, d_base = curvature_baseline(data)
    m = all_metrics(x, d_pred, data["d_true_dense"])
    mb = all_metrics(x, d_base, data["d_true_dense"])
    krep = min(5, len(data["loads"]) - 1)
    names = ["sin 1", "sin 2", "sin 3", "sin 4", "patch@0.33", "patch@0.66"]
    defl = dict(load_name=names[krep], x=x.tolist(),
                w_true=np.asarray(data["w_true_dense"][krep]).tolist(),
                w_pred=pinn.predict_deflection(krep, x).tolist(),
                sensor_x=data["sensor_x"].tolist(),
                sensor_w=np.asarray(data["measurements"][krep]).tolist())
    return dict(sensor_x=data["sensor_x"].tolist(), x=x.tolist(),
                d_true=data["d_true_dense"].tolist(), d_pred=d_pred.tolist(),
                d_baseline=d_base.tolist(), EI_true=data["EI_true_dense"].tolist(),
                EI_pred=EI_pred.tolist(), defl=defl, metrics=m, metrics_baseline=mb,
                n_sensors=int(data["n_sensors"]), noise_frac=float(noise_frac))


def _agg(dicts, key):
    return {k: (float(np.mean([d[key][k] for d in dicts])),
               float(np.std([d[key][k] for d in dicts]))) for k in dicts[0][key]}


def run_scenario_mc(segs, n_sensors, noise_frac, cfg_kwargs, seeds, bc="ss", save_to=None):
    """Monte-Carlo over noise seeds; returns mean reconstruction + uncertainty band."""
    runs = [run_one(segs, n_sensors, noise_frac, cfg_kwargs, seed=s, bc=bc,
                    save_to=(save_to if i == 0 else None)) for i, s in enumerate(seeds)]
    x = np.array(runs[0]["x"])
    dpreds = np.array([r["d_pred"] for r in runs])          # (S, Nx)
    dbase = np.array([r["d_baseline"] for r in runs])
    mmean = _agg(runs, "metrics")
    mbmean = _agg(runs, "metrics_baseline")
    locs = [r["metrics"]["localization_error_pct"] for r in runs]
    rep = runs[int(np.argmin(np.abs(np.array(locs) - mmean["localization_error_pct"][0])))]
    return dict(
        sensor_x=rep["sensor_x"], x=runs[0]["x"], d_true=rep["d_true"],
        EI_true=rep["EI_true"], EI_pred=rep["EI_pred"], defl=rep["defl"],
        d_pred=dpreds.mean(0).tolist(), d_pred_std=dpreds.std(0).tolist(),
        d_baseline=dbase.mean(0).tolist(),
        metrics={k: v[0] for k, v in mmean.items()},
        metrics_std={k: v[1] for k, v in mmean.items()},
        metrics_baseline={k: v[0] for k, v in mbmean.items()},
        n_sensors=int(n_sensors), noise_frac=float(noise_frac), n_seeds=len(seeds))


# --------------------------------------------------------------------------- #
#  Detection study                                                            #
# --------------------------------------------------------------------------- #
def _detect_stat(severity, loc, seed, cfg_kwargs, n_sensors=7, noise=0.02, bc="ss"):
    beam = BeamModel(L=1.0, n_elem=200, EI0=1.0, bc=bc)
    segs = [] if severity <= 0 else [seg(loc, 0.04, severity)]
    data = build_dataset(beam, segs, n_sensors=n_sensors, noise_frac=noise, seed=seed)
    cfg = AegisConfig(bc=bc, **cfg_kwargs)
    pinn = AegisPINN(cfg, data["loads"])
    pinn.set_data(data["sensor_x"], data["measurements"], data["w_scale"])
    pinn.train(verbose=False)
    return float(np.max(pinn.predict_damage(data["x_dense"])))


def detection_study(cfg_kwargs, severities, n_trials, n_sensors=7, noise=0.02):
    rng = np.random.default_rng(7)
    healthy = [_detect_stat(0.0, 0.5, 5000 + t, cfg_kwargs, n_sensors, noise)
               for t in range(n_trials)]
    damaged, roc = {}, []
    for sev in severities:
        stats = [_detect_stat(sev, float(rng.uniform(0.25, 0.75)), 6000 + int(sev * 100) * 50 + t,
                              cfg_kwargs, n_sensors, noise) for t in range(n_trials)]
        damaged[int(sev * 100)] = stats
        hi = max(max(healthy), max(stats)) * 1.02 + 1e-6
        thr = np.linspace(0, hi, 200)
        tpr = np.array([np.mean(np.array(stats) >= t) for t in thr])
        fpr = np.array([np.mean(np.array(healthy) >= t) for t in thr])
        _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        auc = float(np.abs(_trapz(tpr, fpr)))
        roc.append(dict(severity=int(sev * 100), fpr=fpr.tolist(), tpr=tpr.tolist(), auc=auc))
        print(f"   detection @ {int(sev*100)}% damage: AUC={auc:.3f} "
              f"(healthy peak~{np.mean(healthy):.3f}, damaged peak~{np.mean(stats):.3f})", flush=True)
    return dict(severities=[int(s * 100) for s in severities], roc=roc,
                healthy_stat=healthy, damaged_stat=damaged)


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        base_cfg = dict(adam_iters=1200, lbfgs_iters=30)
        gallery, seeds, det_sev, det_n = GALLERY[:3], [101, 102], [0.4], 5
    else:
        base_cfg = dict(adam_iters=2500, lbfgs_iters=40)
        gallery, seeds = GALLERY, [101, 102, 103, 104]
        det_sev, det_n = [0.2, 0.4], 14

    print(f"Running {len(gallery)} scenarios x {len(seeds)} seeds "
          f"(adam={base_cfg['adam_iters']})", flush=True)
    t_all = time.time()

    scenarios = []
    for i, (sid, label, cat, segs, nsen, noise) in enumerate(gallery):
        t0 = time.time()
        save_to = os.path.join(MODELS, "aegis_canonical.pt") if sid == "canonical" else None
        r = run_scenario_mc(segs, nsen, noise, dict(**base_cfg), seeds, save_to=save_to)
        r.update(id=sid, label=label, category=cat)
        scenarios.append(r)
        print(f"[{i+1:2d}/{len(gallery)}] {sid:13s} "
              f"loc={r['metrics']['localization_error_pct']:4.1f}+/-{r['metrics_std']['localization_error_pct']:3.1f}%  "
              f"IoU={r['metrics']['detection_iou']:.2f}  "
              f"base={r['metrics_baseline']['localization_error_pct']:4.1f}%  ({time.time()-t0:.0f}s)", flush=True)

    print("Running ablations ...", flush=True)
    abl_segs = [seg(0.60, 0.04, 0.40)]
    ablations = {}
    for name, kw in [("full", {}), ("no_physics", dict(use_physics=False)),
                     ("no_tv", dict(w_tv=0.0, w_l1=0.0))]:
        r = run_one(abl_segs, 7, 0.02, dict(**base_cfg, **kw), seed=101)
        ablations[name] = dict(x=r["x"], d_true=r["d_true"], d_pred=r["d_pred"], metrics=r["metrics"])
    r1 = run_one_singleload(abl_segs, 7, 0.02, dict(**base_cfg))
    ablations["single_load"] = r1
    print(f"   full={ablations['full']['metrics']['localization_error_pct']:.1f}%  "
          f"no_physics={ablations['no_physics']['metrics']['localization_error_pct']:.1f}%  "
          f"no_tv={ablations['no_tv']['metrics']['localization_error_pct']:.1f}%", flush=True)

    print("Running robustness sweeps (Monte-Carlo) ...", flush=True)
    if args.quick:
        sensors_list, noise_list = [5, 7, 11], [0.0, 0.04, 0.08]
    else:
        sensors_list, noise_list = [4, 5, 7, 9, 11, 15], [0.0, 0.02, 0.04, 0.06, 0.08]
    rob = dict(sensors=dict(counts=[], aegis=[], aegis_std=[], baseline=[]),
               noise=dict(levels=[], aegis=[], aegis_std=[], baseline=[]))
    for nsen in sensors_list:
        r = run_scenario_mc(abl_segs, nsen, 0.02, dict(**base_cfg), seeds)
        rob["sensors"]["counts"].append(nsen)
        rob["sensors"]["aegis"].append(r["metrics"]["localization_error_pct"])
        rob["sensors"]["aegis_std"].append(r["metrics_std"]["localization_error_pct"])
        rob["sensors"]["baseline"].append(r["metrics_baseline"]["localization_error_pct"])
        print(f"   sensors={nsen:2d}  aegis={r['metrics']['localization_error_pct']:.1f}"
              f"+/-{r['metrics_std']['localization_error_pct']:.1f}%", flush=True)
    for noise in noise_list:
        r = run_scenario_mc(abl_segs, 7, noise, dict(**base_cfg), seeds)
        rob["noise"]["levels"].append(round(noise * 100, 1))
        rob["noise"]["aegis"].append(r["metrics"]["localization_error_pct"])
        rob["noise"]["aegis_std"].append(r["metrics_std"]["localization_error_pct"])
        rob["noise"]["baseline"].append(r["metrics_baseline"]["localization_error_pct"])
        print(f"   noise={noise*100:.0f}%  aegis={r['metrics']['localization_error_pct']:.1f}"
              f"+/-{r['metrics_std']['localization_error_pct']:.1f}%", flush=True)

    print("Running detection study ...", flush=True)
    detection = detection_study(dict(**base_cfg), det_sev, det_n)

    # ---- assemble output ------------------------------------------------- #
    sct = [s for s in scenarios if s["category"] in ("single", "multi")]
    loc = float(np.mean([s["metrics"]["localization_error_pct"] for s in sct]))
    loc_std = float(np.mean([s["metrics_std"]["localization_error_pct"] for s in sct]))
    base = float(np.mean([s["metrics_baseline"]["localization_error_pct"] for s in sct]))
    best_auc = max(r["auc"] for r in detection["roc"])
    headline = dict(loc_error_pct=loc, loc_error_std=loc_std, n_sensors=7,
                    improvement_factor=base / max(loc, 1e-6), baseline_loc=base,
                    detection_auc=best_auc, detection_severity=detection["roc"][-1]["severity"])
    table = [dict(scenario=s["label"], n_sensors=s["n_sensors"], noise=s["noise_frac"],
                  loc_err=s["metrics"]["localization_error_pct"],
                  loc_err_std=s["metrics_std"]["localization_error_pct"],
                  sev_err=s["metrics"]["severity_error_pct"],
                  iou=s["metrics"]["detection_iou"],
                  fpr=s["metrics"]["false_positive_rate"],
                  loc_err_base=s["metrics_baseline"]["localization_error_pct"]) for s in scenarios]
    figures = [
        dict(file="fig_headline.png", caption="Damage reconstruction with uncertainty band (mean +/- std over noise realizations) vs. ground truth and the classical baseline."),
        dict(file="fig_gallery.png", caption="Reconstructions across single- and multi-damage scenarios."),
        dict(file="fig_robustness.png", caption="Localization error (mean +/- std) vs. sensor count and measurement noise."),
        dict(file="fig_detection.png", caption="Damage-detection ROC curves and AUC (healthy vs. damaged trials)."),
        dict(file="fig_ablation.png", caption="Ablations: removing the physics or the priors degrades recovery."),
        dict(file="fig_deflection.png", caption="Deflection fit and recovered stiffness field for the canonical scenario."),
    ]
    abstract = (
        "Aging beams and bridges accumulate localized stiffness loss long before "
        "visible failure. We present Aegis, a physics-informed neural network that "
        "localizes and quantifies such damage from only a handful of sparse, noisy "
        "deflection measurements by embedding the Euler-Bernoulli beam equation as an "
        "exact, differentiable forward solver. Across single- and multi-damage scenarios "
        f"Aegis localizes damage to within {loc:.1f}+/-{loc_std:.1f}% of span and "
        f"detects {detection['roc'][-1]['severity']}% damage with AUC {best_auc:.2f}, "
        "while a classical curvature baseline fails under the same sparse, noisy data.")

    out = dict(
        meta=dict(title="Aegis", generated=time.strftime("%Y-%m-%d"), L=1.0, EI0=1.0,
                  n_load_cases=6, n_grid=base_cfg.get("n_grid", 400), n_seeds=len(seeds), bc="ss"),
        headline=headline, scenarios=scenarios, ablations=ablations, robustness=rob,
        detection=detection, table=table, figures=figures, abstract=abstract)
    with open(os.path.join(RESULTS, "results_full.json"), "w") as f:
        json.dump(out, f)
    with open(os.path.join(RESULTS, "metrics.json"), "w") as f:
        json.dump(dict(headline=headline, table=table,
                       detection_auc={r["severity"]: r["auc"] for r in detection["roc"]}), f, indent=2)
    with open(os.path.join(WEBDATA, "results.js"), "w") as f:
        f.write("/* Auto-generated by src/run_experiments.py */\nwindow.AEGIS_RESULTS = ")
        json.dump(out, f)
        f.write(";\n")

    print(f"\nDONE in {time.time()-t_all:.0f}s.  loc = {loc:.1f}+/-{loc_std:.1f}%  "
          f"({headline['improvement_factor']:.1f}x better)  detection AUC = {best_auc:.2f}")
    print("wrote results/results_full.json, results/metrics.json, website/data/results.js")


def run_one_singleload(segs, n_sensors, noise_frac, cfg_kwargs, seed=101):
    from beam_simulation import load_library, sample_sensors
    from datagen import damage_field, damage_on_grid
    beam = BeamModel(L=1.0, n_elem=200, EI0=1.0, bc="ss")
    loads = load_library(beam.L, 1.0)[:1]
    rng = np.random.default_rng(seed)
    EI_elem = beam.EI_from_damage(damage_field(beam, segs))
    sensor_x = np.linspace(0, beam.L, n_sensors + 2)[1:-1]
    x = np.linspace(0, beam.L, 400)
    meas, wsc = [], []
    for q in loads:
        u = beam.solve_static(EI_elem, q)
        wsc.append(np.max(np.abs(beam.deflection_at(x, u))) + 1e-12)
        meas.append(sample_sensors(beam, u, sensor_x, noise_frac, rng))
    cfg = AegisConfig(**cfg_kwargs)
    pinn = AegisPINN(cfg, loads)
    pinn.set_data(sensor_x, meas, np.array(wsc))
    pinn.train(verbose=False)
    d_pred = pinn.predict_damage(x)
    d_true = damage_on_grid(x, segs, beam.L)
    return dict(x=x.tolist(), d_true=d_true.tolist(), d_pred=d_pred.tolist(),
                metrics=all_metrics(x, d_pred, d_true))


if __name__ == "__main__":
    main()
