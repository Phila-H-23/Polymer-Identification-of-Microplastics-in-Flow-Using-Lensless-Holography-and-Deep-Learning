"""Build results/summary_v2.csv for the channel-mode grid (task 3 of the
stokes/channel-mode instruction): one row per run of
  compact CNN x {stokes,pol4,intensity} x {seed 0,1,2}  (9 runs)
  resnet18    x stokes                  x {seed 0}      (1 run)
plus a grouped mean +/- std table of test macro-F1 by (architecture, channels).

wall_time_sec values are taken from each run's own $start/$end PowerShell timer
(ELAPSED_SECONDS), captured from the background task outputs at launch time,
since neither training script logs wall time itself.
"""
import json
import csv
from pathlib import Path

RUNS = [
    ("results/compact_particle_stokes_s0", "compact", "stokes", 0, 928.4),
    ("results/compact_particle_pol4_s0", "compact", "pol4", 0, 2136.7481464),
    ("results/compact_particle_intensity_s0", "compact", "intensity", 0, 2509.4672587),
    ("results/compact_particle_stokes_s1", "compact", "stokes", 1, 3414.0631309),
    ("results/compact_particle_pol4_s1", "compact", "pol4", 1, 7949.2158525),
    ("results/compact_particle_intensity_s1", "compact", "intensity", 1, 2391.9054197),
    ("results/compact_particle_stokes_s2", "compact", "stokes", 2, 674.4037056),
    ("results/compact_particle_pol4_s2", "compact", "pol4", 2, 2523.3118505),
    ("results/compact_particle_intensity_s2", "compact", "intensity", 2, 2865.0737837),
    ("results/resnet18_particle_stokes_s0", "resnet18", "stokes", 0, 502.1413811),
]

CLASSES = ["PE", "PET", "PMMA", "PP", "PS", "PVA", "PVC"]


def load_run(out_dir, arch, channels, seed, wall_time):
    d = json.load(open(Path(out_dir) / "results.json"))
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
    }
    for c in CLASSES:
        row[f"recall_{c}"] = round(test_report.get(c, {}).get("recall", float("nan")), 4)
    return row


def main():
    rows = [load_run(*r) for r in RUNS]

    cols = ["tag", "architecture", "channels", "seed", "n_params", "epochs_run",
            "best_epoch", "best_val_macro_f1", "final_train_macro_f1",
            "final_val_macro_f1", "test_macro_f1", "test_accuracy", "wall_time_sec"]
    cols += [f"recall_{c}" for c in CLASSES]

    out_path = Path("results/summary_v2.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"saved -> {out_path}")
    print()
    print(f"{'tag':38s} {'arch':9s} {'channels':10s} {'seed':4s} {'test_macro_f1':>13s} {'test_acc':>9s} {'wall_s':>8s}")
    for r in rows:
        print(f"{r['tag']:38s} {r['architecture']:9s} {r['channels']:10s} {r['seed']:<4d} "
              f"{r['test_macro_f1']:>13.4f} {r['test_accuracy']:>9.4f} "
              f"{(r['wall_time_sec'] if r['wall_time_sec'] != '' else float('nan')):>8}")

    # grouped mean +/- std by (architecture, channels)
    groups = {}
    for r in rows:
        key = (r["architecture"], r["channels"])
        groups.setdefault(key, []).append(r["test_macro_f1"])

    print()
    print("=== mean +/- std test macro-F1 by (architecture, channels) ===")
    print(f"{'architecture':12s} {'channels':10s} {'n':3s} {'mean':>8s} {'std':>8s}  values")
    grouped_rows = []
    for (arch, ch), vals in sorted(groups.items()):
        n = len(vals)
        mean = sum(vals) / n
        std = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5 if n > 1 else 0.0
        grouped_rows.append({"architecture": arch, "channels": ch, "n": n,
                              "mean_test_macro_f1": round(mean, 4),
                              "std_test_macro_f1": round(std, 4)})
        print(f"{arch:12s} {ch:10s} {n:<3d} {mean:>8.4f} {std:>8.4f}  {[round(v,4) for v in vals]}")

    with open(out_path, "a", newline="") as f:
        f.write("\n")
        w = csv.DictWriter(f, fieldnames=["architecture", "channels", "n", "mean_test_macro_f1", "std_test_macro_f1"])
        w.writeheader()
        for gr in grouped_rows:
            w.writerow(gr)

    # verdict for compact CNN
    compact_groups = {ch: v for (arch, ch), v in groups.items() if arch == "compact"}
    means = {ch: sum(v) / len(v) for ch, v in compact_groups.items()}
    stds = {ch: (sum((x - means[ch]) ** 2 for x in v) / len(v)) ** 0.5 for ch, v in compact_groups.items()}
    best_ch = max(means, key=means.get)
    print()
    print(f"Compact CNN: best channel mode = {best_ch} (mean test macro-F1 = {means[best_ch]:.4f})")
    for ch in sorted(means, key=means.get, reverse=True):
        print(f"  {ch:10s} mean={means[ch]:.4f} std={stds[ch]:.4f}")
    others = [ch for ch in means if ch != best_ch]
    gaps = [means[best_ch] - means[ch] for ch in others]
    max_relevant_std = max(stds[best_ch], *[stds[ch] for ch in others])
    exceeds = all(gap > max_relevant_std for gap in gaps)
    print(f"Gap to next-best exceeds max(std) of the two groups being compared: {exceeds}")


if __name__ == "__main__":
    main()
