"""Rewrites results/report_recon_check.txt: the earlier sections unchanged (recon_check.report), then the follow-up
sections from the follow-up outputs in the scratchpad (partB2.json, partC_*.json, render.json, ratios.json,
newvids.json, followup_waits.json, followup_status.log) and the readings written after viewing the sheets
(readings.json). No em dashes, no use of the word that the task forbids."""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import recon_check as rc  # noqa: E402

W = rc.WORK


def j(name):
    return rc.rjson(name)


def main():
    base = rc.report(write=False).split("\n")
    L = list(base)
    sel = j("selection.json")["selection"]
    B2, R, NV, RD = j("partB2.json"), j("ratios.json") or {}, j("newvids.json") or {}, j("readings.json") or {}
    PC = {s["i"]: j(f"partC_{s['i']}.json") for s in sel}
    # sharpness metrics recomputed from the saved panels: columns with any pixel outside the sensor are dropped first,
    # then rows (the Part C run dropped rows first, which emptied the two frame-edge patches)
    from reconstruction_zsweep2 import laplacian_variance_amp, tamura_coefficient
    for s_ in sel:
        if PC.get(s_["i"]) is None:
            continue
        npz = np.load(os.path.join(W, f"partC_{s_['i']}.npz"))
        for key in list(PC[s_["i"]]["metrics"]):
            img = npz[key]
            img = img[:, np.isfinite(img).all(axis=0)]
            img = img[np.isfinite(img).all(axis=1)]
            a = np.sqrt(np.clip(img, 0, None))
            PC[s_["i"]]["metrics"][key] = dict(laplacian_var=float(laplacian_variance_amp(a)), tamura=float(tamura_coefficient(a)))
    status = open(os.path.join(W, "followup_status.log"), encoding="utf-8").read().splitlines() \
        if os.path.exists(os.path.join(W, "followup_status.log")) else []

    L += ["", "=" * 110, "FOLLOW-UP (requested 2026-10-07 after the report above; same rules)", "=" * 110]
    # ---------------------------------------------------------------- Part B gate change
    L += ["", "PART B GATE CHANGED, ON THE RECORD",
          "Decision by Phila and Fraser after seeing the result above: the 0.05 limit on the zero-phase inside amplitude",
          "tested the field model, not the propagator; the excess (0.554 to 0.556) is the same for all five variants and",
          "cannot favour any of them. New gate: PASS if the exact-phase round trip error is below 1e-5 and the radius error",
          "is at most 10% in every combination; the zero-phase inside amplitude is reported, not gated. The original result",
          "(FAILED the old gate) stays above unchanged. Round trip error measured, as fixed before computing, as the largest",
          "|recovered amplitude - true amplitude| within r <= 4R of the disc centre (the full-grid value is also listed;",
          "with reflect padding the mirrored complex field disturbs only the outer edge of the grid)."]
    if B2:
        L.append(f"{'disc':48s} {'dx um':>5s} {'method':62s} {'inside zp':>9s} {'R err zp':>8s} {'round trip r<=4R':>16s} "
                 f"{'full grid':>9s} {'old gate':>8s} {'new gate':>8s}")
        for r in B2["rows"]:
            L.append(f"{r['disc']:48s} {r['dx_um']:5.2f} {r['method']:62s} {r['inside_zero_phase']:9.4f} "
                     f"{r['radius_err_zero_phase']:+8.1%} {r['exact_round_trip_err_region']:16.2e} {r['exact_round_trip_err_full']:9.2e} "
                     f"{'pass' if r['old_gate_pass'] else 'FAIL':>8s} {'pass' if r['new_gate_pass'] else 'FAIL':>8s}")
        L.append(f"Under the new gate, Part B (50 um disc, every combination): {'PASSED' if B2['new_gate_all_pass'] else 'FAILED'}; "
                 f"under the old gate: {'passed' if B2['old_gate_all_pass'] else 'failed'}.")
        bad = [r for r in B2["rows"] if r["disc"].startswith("50") and not r["new_gate_pass"]]
        if bad:
            L.append("Failing the new gate: " + "; ".join(f"{r['method']} at {r['dx_um']:.2f} um (round trip "
                                                            f"{r['exact_round_trip_err_region']:.3e})" for r in bad) +
                     ". These are reflect-padded set-ups at 6.9 um, which no variant uses (V1 is unpadded at 6.9 um, V2 to V5 "
                     "run at 3.45 um, where the padded round trip is 4.1e-6). Recorded as a mechanical failure of the fixed "
                     "rule; the rule is not rewritten; Part C was run because the follow-up and the overnight order said so.")
        old = [r for r in B2["rows"] if r["disc"].startswith("138") and r["method"].startswith("build") and "unpadded" in r["method"]]
        if old:
            o = old[0]
            L.append(f"Why 0.520 then and 0.554 to 0.556 now: the 2026-09-30 check (recon_validate.log) used a disc of radius 20 px = "
                     f"138 um at 6.9 um, this run the 50 um radius disc of the z-sweep test; pitch (6.9 um), grid (512), z, wavelength, "
                     f"amplitudes (0.5 / 1.0), no noise and the inside measure (mean over r <= R) are the same, and the 138 um disc "
                     f"through this run's code gives {o['inside_zero_phase']:.4f}, so the disc size is the difference (Fresnel number "
                     f"a^2 / (lambda z) 27.8 for 138 um, 3.6 for 50 um: the twin image overlaps the smaller disc more).")
    # ---------------------------------------------------------------- Part C
    L += ["", "PART C  CONTACT SHEETS (run after the gate change)"]
    waits = j("followup_waits.json") or {}
    L.append("Run conditions: background mode (idle CPU, low I/O and memory priority), one thread; nothing ran before the z0 "
             "build finished; paused during the chain's collisions step, during its first (timed) z0 run and whenever the chain "
             "would have kept less than 1 GB of available RAM. Raw frames read in column bands (HDF5) or one streamed zlib pass "
             "(PET4) into disk-backed stacks.")
    if waits:
        L.append("Waiting time per step (min): " + ", ".join(f"{k} {v / 60:.1f}" for k, v in waits.items()))
    rp = [s for s in status if "read pass" in s]
    if rp:
        L.append("Read passes: " + "; ".join(s.split("read pass ")[1] for s in rp))
    L.append(f"{'#':>2s} {'patch':30s} {'bg frames':>9s} {'backgrounds s':>13s} {'V1 s':>6s} {'V2+V5 s':>8s} {'V3+V4 s':>8s} "
             f"{'total min':>9s} {'V1 vs stored recon_s0':>22s}")
    for s in sel:
        p = PC.get(s["i"])
        if not p:
            L.append(f"{s['i']:>2d} {s['patch_file'][:-4]:30s} not run")
            continue
        T = p["timings"]
        L.append(f"{s['i']:>2d} {s['patch_file'][:-4]:30s} {p['n_bg_frames']:>9d} {T['backgrounds']:>13.0f} {T['V1']:>6.0f} "
                 f"{T['V2 (41 z) and V5 (3 z)']:>8.0f} {T['V3 and V4']:>8.0f} {T['total'] / 60:>9.1f} "
                 f"{p.get('V1_vs_stored_recon_s0_max_abs_diff', float('nan')):>22.2e}")
    L += ["V1 is the build code called unchanged; its z = 2.25 mm output equals the stored recon_s0 patches to the listed",
          "difference. Display: every intensity panel of a patch uses one range, the 1st to 99th percentile of the pooled",
          "pixels of that patch's 15 sheet-2 panels (rule fixed before rendering); subtraction variants are divided by their",
          "scalar background mean (V2, V5: mean(B); V4: mean over channels of mean(B_c)), V1 and V3 are ratios, the raw I0",
          "panel is I0 divided by the frame's median I0. Retardance and slow axis: recon.birefringence_eta_theta on the four",
          "propagated channels, converted to radians (eta x pi/2; theta x pi wrapped to -pi/2..pi/2), without the per-video",
          "A_bg / B_bg maps, for V1 and V2 alike. Blue = outside the sensor.",
          "Files: results/recon_check/S1_V2_zsweep_<#>_<patch>.png, S2_variants_<#>_<patch>.png, S3_birefringence_<#>_<patch>.png."]
    rr = j("render.json")
    if rr:
        L.append("Display range per patch: " + "; ".join(f"#{k} {v[0]:.3f} to {v[1]:.3f}" for k, v in rr["ranges"].items()))
    # metrics
    if all(PC.get(s["i"]) for s in sel):
        L += ["", "Sharpness metrics (code of the earlier z sweep: Laplacian variance and Tamura coefficient of the amplitude,",
              "sqrt of the shown intensity, over the panel's in-sensor pixels). Caveat: on synthetic discs these metrics were",
              "flat across z (recon_zsweep2), so they are listed for completeness, not as a focus criterion. Laplacian values",
              "depend on the pixel size, so 3.45 um panels (V3, V4) and 6.9 um panels (V1, V2, V5) are not comparable."]
        for metric, name in (("laplacian_var", "Laplacian variance"), ("tamura", "Tamura coefficient")):
            L.append(f"Sheet 1 (V2), {name}, rows z in mm, columns patch #:")
            L.append(f"{'z':>6s} " + " ".join(f"{('#' + str(s['i'])):>10s}" for s in sel))
            for z in rc.Z41:
                k = f"V2_{z * 1e3:.3f}"
                L.append(f"{z * 1e3:6.3f} " + " ".join(f"{PC[s['i']]['metrics'][k][metric]:10.3e}" for s in sel))
            best = []
            for s in sel:
                vals = [PC[s["i"]]["metrics"][f"V2_{z * 1e3:.3f}"][metric] for z in rc.Z41]
                best.append(f"#{s['i']} {rc.Z41[int(np.nanargmax(vals))] * 1e3:.3f}")
            L.append(f"  z of the largest value per patch: " + ", ".join(best))
        for metric, name in (("laplacian_var", "Laplacian variance"), ("tamura", "Tamura coefficient")):
            L.append(f"Sheet 2, {name}, rows variant and z, columns patch #:")
            L.append(f"{'':10s} " + " ".join(f"{('#' + str(s['i'])):>10s}" for s in sel))
            for v in ("V1", "V2", "V3", "V4", "V5"):
                for z in rc.Z3:
                    k = f"{v}_{z * 1e3:.3f}"
                    L.append(f"{v} {z * 1e3:5.2f}  " + " ".join(f"{PC[s['i']]['metrics'][k][metric]:10.3e}" for s in sel))
    # readings
    L += ["", "READINGS (fixed in the task; description of what is visible, not a judgement of which chain is better)"]
    if RD:
        L.append(f"Sheet 1 reading: {RD['D']}")
        for line in RD.get("D_detail", []):
            L.append(f"  {line}")
        L.append("Variants at z = 2.25 mm (sheet 2), V2 to V5 compared with V1, one sentence per patch:")
        for s in sel:
            L.append(f"  #{s['i']} {s['patch_file'][:-4]}: {RD['variants'][str(s['i'])]}")
        for line in RD.get("other", []):
            L.append(line)
    else:
        L.append("(written after the sheets have been viewed)")
    # ---------------------------------------------------------------- I90 / I0
    L += ["", "I90 / I0 RECONCILIATION",
          "Mosaic labels (recon.demosaic, 0-based rows and columns of the 2048 x 2448 frame as returned by get_raw_frame; HDF5",
          "frames transposed from the on-disk 2448 x 2048 order, MAT v5 frames used as stored): I90 = [even row, even column],",
          "I45 = [even row, odd column], I135 = [odd row, even column], I0 = [odd row, odd column]; in MATLAB terms the 2 x 2",
          "cell reads 90 45 / 135 0 (I90 at (1,1)), as confirmed with Fraser and used by every build.",
          "Background frames per recording: 30 evenly spaced frames over the whole recording, per-pixel median over them (moving",
          "particles drop out), then: (a) median(I90) / median(I0) over the sensor, (b) median over pixels of I90 / I0, plus the",
          "ratio of means and the Stokes q of to_stokes / stokes_diagnostic_v2 (q = (I0 - I90) / (S + 0.05), S = mean of the",
          "four channels, values / 255). 'after offset removal': no build subtracts a camera dark or black level; the builds",
          "change the channel balance only through (i) build_bgcorrected_dataset.correct_frame (raw - B + mean(B), B = per-pixel",
          "mean of up to 90 preceding frames, mean(B) one scalar over the mosaic) and (ii) the z0 / reconstruction division by",
          "the per-channel median background. Both are applied to each Part C patch frame below (its own 90-frame window).",
          f"{'recording':12s} {'frames':>6s} {'med I0':>7s} {'med I45':>7s} {'med I90':>7s} {'med I135':>8s} {'(a) ratio':>9s} "
          f"{'(b) ratio':>9s} {'ratio means':>11s} {'q stokes':>8s}"]
    for name in ("PET1", "PET2", "PET3", "PET4", "PS1", "PS2", "frames_bg_30"):
        r = R.get(name)
        if not r:
            continue
        x = r["raw"]
        L.append(f"{name:12s} {r['n_frames']:>6d} {x['med_I0']:7.1f} {x['med_I45']:7.1f} {x['med_I90']:7.1f} {x['med_I135']:8.1f} "
                 f"{x['ratio_of_medians']:9.3f} {x['median_of_ratio']:9.3f} {x['ratio_of_means']:11.3f} {x['q_stokes_median']:8.3f}")
    if "frames_bg_30" in R:
        L.append(f"frames_bg_30.mat, q_ref as computed by stokes_diagnostic_v2 (mean of 30 frames, demosaic, / 255, mean of the "
                 f"q map): {R['frames_bg_30']['q_ref_mean_as_stokes_diagnostic_v2']:.3f}")
    L.append("Per Part C frame (raw uint8 / after the bgcorrected build's correction / after the z0 division):")
    L.append(f"{'#':>2s} {'patch':30s} {'(a) raw':>8s} {'(b) raw':>8s} {'(a) bgcorr':>10s} {'(b) bgcorr':>10s} {'(a) z0':>7s} {'(b) z0':>7s}")
    for s in sel:
        p = PC.get(s["i"])
        if p and "ratios_frame" in p:
            f_ = p["ratios_frame"]
            L.append(f"{s['i']:>2d} {s['patch_file'][:-4]:30s} {f_['raw']['ratio_of_medians']:8.3f} {f_['raw']['median_of_ratio']:8.3f} "
                     f"{f_['bgcorrected']['ratio_of_medians']:10.3f} {f_['bgcorrected']['median_of_ratio']:10.3f} "
                     f"{f_['z0_divided']['ratio_of_medians']:7.3f} {f_['z0_divided']['median_of_ratio']:7.3f}")
    L += [RD.get("ratio_conclusion", "(conclusion written after the numbers are in)")]
    # ---------------------------------------------------------------- PET4 vs PET3
    L += ["", "PET4 FORMAT CHECK (PET4 against PET3; metadata from the earlier extras plus the 30-frame backgrounds)"]
    E = {v["video"]: v for v in j("extras.json")["videos"]}
    for v in ("PET3", "PET4"):
        e, r = E[v], R.get(v)
        if not r:
            continue
        shape = e.get("shape_on_disk") or e["variables"][0][1]
        L.append(f"{v}: {e['container']}; header '{e['header_text']}'; variable {e.get('dataset') or e['variables'][0][0]}; "
                 f"array on disk {shape}; dtype uint8; frames {r['n_frames']}; frame 2048 x 2448; file {e['size_bytes'] / 1e9:.2f} GB")
        lv = r["levels"]
        L.append("    background per channel, mean (spatial std; temporal std of frame means; P5 to P95): " + "; ".join(
            f"{c} {lv[c]['mean']:.1f} ({lv[c]['spatial_std']:.1f}; {lv[c]['temporal_std_of_frame_means']:.2f}; "
            f"{lv[c]['p5']:.0f} to {lv[c]['p95']:.0f})" for c in ("I0", "I45", "I90", "I135")))
        L.append(f"    I90 / I0: (a) {r['raw']['ratio_of_medians']:.3f}, (b) {r['raw']['median_of_ratio']:.3f}")
    L += [RD.get("pet4_conclusion", "(conclusion written after the numbers are in)")]
    # ---------------------------------------------------------------- new videos
    L += ["", "NEW VIDEOS: PE500FRAMES AND PP500FRAMES (no patches, no manifests)",
          "The earlier addendum about these two files was not run before (it is not in this session's task text); run now after",
          "Part C with the same safety rules and band reading. Detections: holo_pipeline_starter.detect_particles with its",
          "default thresholds (z > 7, 10 px dilation, 500 to 60000 px), and the count left after the Step 6 debris check",
          "(comparison frame 10 frames later), on 20 evenly spaced frames; background from the per-pixel median of the 20 frames."]
    for name in ("PE500FRAMES", "PP500FRAMES"):
        n = NV.get(name)
        if not n:
            L.append(f"{name}: not run")
            continue
        L.append(f"{name}.mat: {n['container']}; header '{n['header_text']}'; file modified {n['file_modified']} (copy date); "
                 f"variable {n['variable']}; array on disk {'x'.join(map(str, n['shape_on_disk']))}; {n['dtype']}; chunks {n['chunks']}; "
                 f"compression {n['compression']}; frames {n['n_frames']}; frame 2048 x 2448")
        lv = n["levels"]
        L.append("    background per channel, mean (spatial std; temporal std of frame means): " + "; ".join(
            f"{c} {lv[c]['mean']:.1f} ({lv[c]['spatial_std']:.1f}; {lv[c]['temporal_std_of_frame_means']:.2f})"
            for c in ("I0", "I45", "I90", "I135")))
        L.append(f"    I90 / I0: (a) {n['raw']['ratio_of_medians']:.3f}, (b) {n['raw']['median_of_ratio']:.3f}; q stokes "
                 f"{n['raw']['q_stokes_median']:.3f}")
        L.append("    frame: regions / detections / kept after debris check: " + "; ".join(
            f"{d['frame']}: {d['n_regions']}/{d['n_detections']}/{d['n_kept_after_debris_check']}" for d in n["detections"]))
        det = np.array([d["n_detections"] for d in n["detections"]])
        kept = np.array([d["n_kept_after_debris_check"] for d in n["detections"]])
        L.append(f"    detections per frame median {np.median(det):.0f} (range {det.min()} to {det.max()}); kept after the debris "
                 f"check median {np.median(kept):.0f} (range {kept.min()} to {kept.max()})")
    L += [RD.get("newvids_conclusion", "")]
    L += ["", "FIVE-LINE SUMMARY (follow-up)"] + RD.get("summary", ["(written at the end)"])
    text = "\n".join(L)
    assert "—" not in text and "significant" not in text.lower()
    open(rc.REPORT, "w", encoding="utf-8").write(text + "\n")
    print(text[-6000:])


if __name__ == "__main__":
    main()
