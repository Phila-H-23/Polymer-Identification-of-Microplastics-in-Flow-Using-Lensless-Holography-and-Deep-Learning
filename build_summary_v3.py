"""Build results/summary_v3.csv after the stokes-formula fix.

Includes, per (architecture, channels):
  compact  stokes     seeds 0,1,2   (re-run with fixed to_stokes())
  compact  pol4       seeds 0-4     (0-2 from grid v2, 3-4 new)
  compact  intensity  seeds 0-4     (0-2 from grid v2, 3-4 new)
  resnet18 pol4       seeds 0-4     (0 from grid v1 @ script defaults, 1-4 new @ same defaults)
  resnet18 intensity  seeds 0-4     (0 from grid v1 @ script defaults, 1-4 new @ same defaults)
  resnet18 stokes     seed 0 only   (NOT re-run - still uses the OLD broken formula;
                                      no re-run was requested for this cell, so it is
                                      flagged rather than silently mixed in as comparable)

wall_time_sec: each run's own $start/$end PowerShell timer, captured from task
outputs at launch time (neither script logs wall time itself).
"""
import json
import csv
from pathlib import Path

CLASSES = ["PE", "PET", "PMMA", "PP", "PS", "PVA", "PVC"]

# (out_dir, architecture, channels, seed, wall_time_sec, note)
RUNS = [
    ("results/compact_particle_stokes_s0", "compact", "stokes", 0, 1404.8309027, ""),
    ("results/compact_particle_stokes_s1", "compact", "stokes", 1, 2115.8623681, ""),
    ("results/compact_particle_stokes_s2", "compact", "stokes", 2, 1671.3234836, ""),

    ("results/compact_particle_pol4_s0", "compact", "pol4", 0, 2136.7481464, ""),
    ("results/compact_particle_pol4_s1", "compact", "pol4", 1, 7949.2158525, ""),
    ("results/compact_particle_pol4_s2", "compact", "pol4", 2, 2523.3118505, ""),
    ("results/compact_particle_pol4_s3", "compact", "pol4", 3, 881.6226299, ""),
    ("results/compact_particle_pol4_s4", "compact", "pol4", 4, 3111.4511161, ""),

    ("results/compact_particle_intensity_s0", "compact", "intensity", 0, 2509.4672587, ""),
    ("results/compact_particle_intensity_s1", "compact", "intensity", 1, 2391.9054197, ""),
    ("results/compact_particle_intensity_s2", "compact", "intensity", 2, 2865.0737837, ""),
    ("results/compact_particle_intensity_s3", "compact", "intensity", 3, 2856.2748421, ""),
    ("results/compact_particle_intensity_s4", "compact", "intensity", 4, 1982.1961911, ""),

    ("results/resnet18_particle_pol4_s0", "resnet18", "pol4", 0, 559.2, ""),
    ("results/resnet18_particle_pol4_s1", "resnet18", "pol4", 1, 798.1501004, ""),
    ("results/resnet18_particle_pol4_s2", "resnet18", "pol4", 2, 871.3169376, ""),
    ("results/resnet18_particle_pol4_s3", "resnet18", "pol4", 3, 760.7484464, ""),
    ("results/resnet18_particle_pol4_s4", "resnet18", "pol4", 4, 18196.6811148, ""),

    ("results/resnet18_particle_intensity_s0", "resnet18", "intensity", 0, 840.8, ""),
    ("results/resnet18_particle_intensity_s1", "resnet18", "intensity", 1, 1010.6584685, ""),
    ("results/resnet18_particle_intensity_s2", "resnet18", "intensity", 2, 1554.0289751, ""),
    ("results/resnet18_particle_intensity_s3", "resnet18", "intensity", 3, 2579.9758861, ""),
    ("results/resnet18_particle_intensity_s4", "resnet18", "intensity", 4, 253.6936507, ""),

    ("results/resnet18_particle_stokes_s0", "resnet18", "stokes", 0, 502.1413811,
     "OLD BROKEN FORMULA - not re-run, not comparable to fixed compact-stokes numbers"),
]


def load_run(out_dir, arch, channels, seed, wall_time, note):
    p = Path(out_dir) / "results.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    n_params = d.get("n_params", d.get("trainable_params"))
    epochs_run = len(d["history"])
    best_epoch = d["best_epoch"]
    best_val_macro_f1 = d["best_val_macro_f1"]
    final_train_macro_f1 = d["history"][-1]["train_macro_f1"]
    final_val_macro_f1 = d["history"][-1]["val_macro_f1"]
    test_report = d["test_report"]
    test_macro_f1 = d.get("test_macro_f1", test_report["macro avg"]["f1-score"])
    test_accuracy = test_report["accuracy"]
    row = {
        "tag": Path(out_dir).name,
        "architecture": arch,
        "channels": channels,
        "seed": seed,
        "n_params": n_params,
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_val_macro_f1, 4),
        "final_train_macro_f1": round(final_train_macro_f1, 4),
        "final_val_macro_f1": round(final_val_macro_f1, 4),
        "test_macro_f1": round(test_macro_f1, 4),
        "test_accuracy": round(test_accuracy, 4),
        "wall_time_sec": round(wall_time, 1) if wall_time is not None else "",
        "note": note,
    }
    for c in CLASSES:
        row[f"recall_{c}"] = round(test_report.get(c, {}).get("recall", float("nan")), 4)
    return row


def main():
    rows = [r for r in (load_run(*spec) for spec in RUNS) if r is not None]
    missing = [spec[0] for spec in RUNS if not (Path(spec[0]) / "results.json").exists()]
    if missing:
        print(f"NOTE: {len(missing)} run(s) not yet available, excluded from this build: {missing}")

    cols = ["tag", "architecture", "channels", "seed", "n_params", "epochs_run",
            "best_epoch", "best_val_macro_f1", "final_train_macro_f1",
            "final_val_macro_f1", "test_macro_f1", "test_accuracy", "wall_time_sec", "note"]
    cols += [f"recall_{c}" for c in CLASSES]

    out_path = Path("results/summary_v3.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"saved -> {out_path}  ({len(rows)} runs)")
    print()
    print(f"{'tag':38s} {'arch':9s} {'channels':10s} {'seed':4s} {'test_macro_f1':>13s} {'test_acc':>9s} {'wall_s':>8s}")
    for r in rows:
        wt = r['wall_time_sec'] if r['wall_time_sec'] != '' else float('nan')
        print(f"{r['tag']:38s} {r['architecture']:9s} {r['channels']:10s} {r['seed']:<4d} "
              f"{r['test_macro_f1']:>13.4f} {r['test_accuracy']:>9.4f} {wt:>8}")

    # grouped mean +/- std by (architecture, channels) - excludes the flagged
    # not-re-run resnet-stokes cell from being treated as a clean comparison point
    groups = {}
    for r in rows:
        key = (r["architecture"], r["channels"])
        groups.setdefault(key, []).append((r["test_macro_f1"], r["note"]))

    print()
    print("=== mean +/- std test macro-F1 by (architecture, channels) ===")
    print(f"{'architecture':12s} {'channels':10s} {'n':3s} {'mean':>8s} {'std':>8s}  values")
    grouped_rows = []
    for (arch, ch), pairs in sorted(groups.items()):
        vals = [v for v, _ in pairs]
        n = len(vals)
        mean = sum(vals) / n
        std = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5 if n > 1 else 0.0
        flag = " [FLAGGED]" if any(note for _, note in pairs) else ""
        grouped_rows.append({"architecture": arch, "channels": ch, "n": n,
                              "mean_test_macro_f1": round(mean, 4),
                              "std_test_macro_f1": round(std, 4)})
        print(f"{arch:12s} {ch:10s} {n:<3d} {mean:>8.4f} {std:>8.4f}  {[round(v,4) for v in vals]}{flag}")

    with open(out_path, "a", newline="") as f:
        f.write("\n")
        w = csv.DictWriter(f, fieldnames=["architecture", "channels", "n", "mean_test_macro_f1", "std_test_macro_f1"])
        w.writeheader()
        for gr in grouped_rows:
            w.writerow(gr)

    # sanity condition: stokes >= intensity for the compact CNN
    compact_groups = {ch: [v for v, _ in pairs] for (arch, ch), pairs in groups.items() if arch == "compact"}
    if "stokes" in compact_groups and "intensity" in compact_groups:
        stokes_mean = sum(compact_groups["stokes"]) / len(compact_groups["stokes"])
        intensity_mean = sum(compact_groups["intensity"]) / len(compact_groups["intensity"])
        print()
        print(f"Sanity condition (compact CNN): stokes mean={stokes_mean:.4f} vs intensity mean={intensity_mean:.4f} "
              f"-> stokes >= intensity: {stokes_mean >= intensity_mean}")

    means = {ch: sum(v)/len(v) for ch, v in compact_groups.items()}
    if means:
        best_ch = max(means, key=means.get)
        print(f"Compact CNN: best channel mode = {best_ch} (mean test macro-F1 = {means[best_ch]:.4f})")


if __name__ == "__main__":
    main()
