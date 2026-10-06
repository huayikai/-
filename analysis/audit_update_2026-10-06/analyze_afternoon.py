"""Compare the four Oct 5 afternoon candidates at the fixed 5M budget.

Input is details.json.gz from src/analysis/analyze_experiments.py. Source
verification compares with the supplied checkout at analysis time; subsequent
output-path edits are intentionally outside the training snapshot comparison.
"""
import argparse
import gzip
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path, PurePosixPath

import numpy as np


def mean_window(series, start, end):
    if not series or series[-1][0] < end:
        return None
    if any(not math.isfinite(y) for _, y in series):
        raise ValueError("nonfinite scalar")
    xs, ys = np.asarray(series, dtype=float).T
    # The first evaluation is after one rollout. Only the initial [0, first]
    # interval uses the first observed value; all other boundaries interpolate.
    if start != 0 and xs[0] > start:
        return None
    x = np.r_[start, xs[(xs > start) & (xs < end)], end]
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(np.interp(x, xs, ys), x) / (end - start))


def dump(out, name, data):
    (out / name).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                      allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).parent / "details.json.gz")
    parser.add_argument("--source-root", type=Path, default=Path(r"D:\src"))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    out = args.input.parent
    details = json.loads(gzip.decompress(args.input.read_bytes()))
    index = {x["row"]["name"]: x for x in details}
    new = [x for x in details if x["row"]["name"].startswith("audit_router_both_mix005_5m_")]
    assert len(new) == 4
    metrics, diffs, verification, diagnostics, scalar_checks = [], [], [], [], []
    diagnostic_tags = ["counterfactual_gate_mean", "counterfactual_gate_std",
                       "counterfactual_gate_target_correlation",
                       "counterfactual_route_target_mean",
                       "counterfactual_route_positive_rate", "counterfactual_route_reliable_rate",
                       "counterfactual_bm_error_mean", "counterfactual_dvd_error_mean",
                       "counterfactual_loss_bm", "counterfactual_loss_dvd",
                       "counterfactual_loss_mix", "counterfactual_loss_value",
                       "adaptive_shift_ratio", "adaptive_bm_grad_norm",
                       "adaptive_dvd_grad_norm", "adaptive_gat_grad_norm", "adaptive_gate_grad_norm"]
    for candidate in new:
        row = candidate["row"]
        map_name, seed = row["map"], row["seed"]
        controls = [("BM", index[f"audit_bm_td_{map_name}_s{seed}"]),
                    ("Router_0.25", index[f"audit_router_both_{map_name}_s{seed}"]),
                    ("Router_0.05", candidate)]
        for method, item in controls:
            r, series = item["row"], item["series"]
            wins = series["test_battle_won_mean"]
            metrics.append({"name": r["name"], "map": map_name, "seed": seed,
                            "method": method, "start_shanghai": r["timestamp"],
                            "n_eval": len(wins), "first_step": wins[0][0],
                            "last_step": wins[-1][0], "t_max": item["config"]["t_max"],
                            "auc_0_5m": mean_window(wins, 0, 5_000_000),
                            "win_4_5m": mean_window(wins, 4_000_000, 5_000_000),
                            "one_m_windows": [mean_window(wins, i*1_000_000, (i+1)*1_000_000) for i in range(5)],
                            "win_9_10m": mean_window(wins, 9_000_000, 10_000_000),
                            "last_evaluation": wins[-1][1],
                            "nonfinite_tags": r["nonfinite_tags"]})
            if method != "BM":
                diagnostics.append({"name": r["name"], "map": map_name, "seed": seed,
                                    "method": method,
                                    "mean_4_5m": {t: mean_window(series[t], 4_000_000, 5_000_000) for t in diagnostic_tags}})
        old_config, new_config = controls[1][1]["config"], candidate["config"]
        difference = {k: {"old_025": old_config.get(k), "new_005": new_config.get(k)}
                      for k in sorted(old_config.keys() | new_config.keys()) if old_config.get(k) != new_config.get(k)}
        assert set(difference) == {"name", "t_max", "counterfactual_mix_loss_weight"}
        diffs.append({"name": row["name"], "differences": difference})
        assert new_config["audit_fix_td_lambda"] is True
        assert new_config["audit_isolate_dvd_aux_hidden"] is True
        assert new_config["counterfactual_mix_loss_weight"] == .05
        run_dir = Path(row["config_path"]).parent
        snapshots = {p.replace("\\", "/"): run_dir.parent / stored
                     for p, stored in candidate["run"]["experiment"]["sources"]}
        for original, stored in candidate["run"]["resources"]:
            relative = original.split("/src/", 1)[1]
            snapshots[relative] = run_dir.parent / "_resources" / PurePosixPath(stored).name
        files = []
        for record in new_config["audit_source_manifest"]:
            rel = record["path"]
            saved = snapshots[rel].read_bytes()
            local = (args.source_root / rel).read_bytes()
            files.append({"path": rel, "snapshot_path": str(snapshots[rel]),
                          "manifest_sha256": record["sha256"],
                          "snapshot_sha256": hashlib.sha256(saved).hexdigest(),
                          "manifest_matches_saved_snapshot": hashlib.sha256(saved).hexdigest() == record["sha256"],
                          "local_matches_ignoring_newlines": saved.replace(b"\r\n", b"\n") == local.replace(b"\r\n", b"\n")})
        verification.append({"name": row["name"], "files": files})
        info = json.loads((run_dir / "info.json").read_text(encoding="utf-8"))
        checked = []
        skipped = []
        for tag, tb in candidate["series"].items():
            if tag not in info or tag + "_T" not in info:
                continue
            if any(not isinstance(v, (int, float)) for v in info[tag]):
                # Sacred stores grad_norm as serialized torch objects. Do not
                # deserialize executable objects just to compare scalars.
                skipped.append(tag)
                continue
            sacred = sorted(dict(zip(info[tag + "_T"], info[tag])).items())
            sacred_by_step, tb_by_step = dict(sacred), dict(tb)
            common = sorted(sacred_by_step.keys() & tb_by_step.keys())
            tb_only = sorted(tb_by_step.keys() - sacred_by_step.keys())
            sacred_only = sorted(sacred_by_step.keys() - tb_by_step.keys())
            same_steps = not tb_only and not sacred_only
            sv = np.asarray([sacred_by_step[s] for s in common], dtype=float)
            tv = np.asarray([tb_by_step[s] for s in common], dtype=float)
            # TensorBoard simple_value stores float32; Sacred stores Python float.
            error = float(np.max(np.abs(sv-tv)))
            close = bool(np.allclose(sv, tv, rtol=2e-7, atol=1e-8))
            checked.append({"tag": tag, "points": len(tb), "same_steps": same_steps,
                            "sacred_points": len(sacred), "common_points": len(common),
                            "tb_only_steps": tb_only, "sacred_only_steps": sacred_only,
                            "values_match_float32_tolerance": close, "max_abs_error": error})
        scalar_checks.append({"name": row["name"], "tags": checked,
                              "skipped_sacred_non_numeric_tags": skipped})
    assert all(r["auc_0_5m"] is not None and not r["nonfinite_tags"] for r in metrics)
    assert all(f["manifest_matches_saved_snapshot"] for r in verification for f in r["files"])
    assert all(t["values_match_float32_tolerance"] for r in scalar_checks for t in r["tags"])
    assert all(s > 5_000_000 for r in scalar_checks for t in r["tags"]
               for s in t["tb_only_steps"] + t["sacred_only_steps"])
    paired = []
    for map_name in sorted({r["map"] for r in metrics}):
        for method in ("BM", "Router_0.25", "Router_0.05"):
            group = [r for r in metrics if r["map"] == map_name and r["method"] == method]
            paired.append({"map": map_name, "method": method, "seeds": [r["seed"] for r in group],
                           "n": len(group),
                           "auc_0_5m": float(np.mean([r["auc_0_5m"] for r in group])),
                           "win_4_5m": float(np.mean([r["win_4_5m"] for r in group]))})
    dump(out, "five_m_metrics.json", metrics)
    dump(out, "paired_metrics.json", paired)
    dump(out, "config_comparison.json", diffs)
    dump(out, "source_verification.json", {"checked_at_local": datetime.now().isoformat(),
                                          "local_code_root": str(args.source_root),
                                          "runs": verification})
    dump(out, "diagnostics.json", diagnostics)
    dump(out, "sacred_tensorboard_checks.json", scalar_checks)
    print(json.dumps({"paired": paired, "source_records": sum(len(x["files"]) for x in verification),
                      "local_source_matches": sum(f["local_matches_ignoring_newlines"] for r in verification for f in r["files"]),
                      "sacred_tb_tags_checked": sum(len(r["tags"]) for r in scalar_checks)}, ensure_ascii=True))
    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 7.6), sharex=True, sharey=True)
        colors = {"BM": "#4d5663", "Router_0.25": "#da8536", "Router_0.05": "#1477bc"}
        for ax, candidate in zip(axes.flat, new):
            r = candidate["row"]
            selected = [x for x in metrics if x["map"] == r["map"] and x["seed"] == r["seed"]]
            for item in selected:
                values = np.asarray(index[item["name"]]["series"]["test_battle_won_mean"], dtype=float)
                values = values[values[:,0] <= 5_000_000]
                x, y = values.T
                method = item["method"]
                ax.plot(x/1e6, y*100, color=colors[method], alpha=.13, linewidth=.55)
                smooth = np.convolve(y, np.ones(21)/21, mode="valid")
                label = method.replace("_", " ") + f" (4-5M: {item['win_4_5m']*100:.1f}%)"
                ax.plot(x[10:-10]/1e6, smooth*100, color=colors[method], linewidth=1.7, label=label)
            ax.axvspan(4, 5, color="#808080", alpha=.06)
            ax.set_title(f"{r['map']} / seed {r['seed']}")
            ax.set_xlim(0, 5)
            ax.set_ylim(0, 100)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8, loc="upper left")
        for ax in axes[-1]:
            ax.set_xlabel("Environment steps (millions)")
        for ax in axes[:,0]:
            ax.set_ylabel("Test win rate (%)")
        fig.suptitle("Oct 5 afternoon: fixed 5M paired comparisons", fontsize=15)
        fig.text(.5, .01, "Faint: raw evaluations. Solid: 21-evaluation moving average. Metrics use unsmoothed time integrals.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0,.025,1,.955))
        fig.savefig(out / "learning_curves.png", dpi=170)
        fig.savefig(out / "learning_curves.svg")
        plt.close(fig)


if __name__ == "__main__":
    main()
