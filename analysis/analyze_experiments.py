"""Read-only audit of Sacred configurations and TensorBoard scalar records."""
import csv
import argparse
import gzip
import json
import math
import re
import struct
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from tensorboard.compat.proto import event_pb2

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, default=Path(r"D:\科研\ablation_results"))
parser.add_argument('--out', type=Path, default=Path(__file__).parent / "experiment_audit")
parser.add_argument('--name-regex', default=None)
args = parser.parse_args()
ROOT, OUT = args.root, args.out
OUT.mkdir(parents=True, exist_ok=True)


def read_events(directory):
    raw = defaultdict(list)
    issues = []
    for path in sorted(directory.glob("events.out*")):
        with path.open("rb") as stream:
            while True:
                header = stream.read(12)
                if not header:
                    break
                if len(header) != 12:
                    issues.append("partial header: " + str(path))
                    break
                length = struct.unpack("<Q", header[:8])[0]
                payload = stream.read(length)
                crc = stream.read(4)
                if len(payload) != length or len(crc) != 4:
                    issues.append("partial record: " + str(path))
                    break
                event = event_pb2.Event()
                event.ParseFromString(payload)
                for value in event.summary.value:
                    if value.HasField("simple_value"):
                        raw[value.tag].append((event.step, event.wall_time, float(value.simple_value)))
                    elif value.HasField("tensor"):
                        tensor = value.tensor
                        if tensor.float_val:
                            raw[value.tag].append((event.step, event.wall_time, float(tensor.float_val[0])))
                        elif tensor.double_val:
                            raw[value.tag].append((event.step, event.wall_time, float(tensor.double_val[0])))
                        else:
                            issues.append("unsupported tensor: " + value.tag)
    series = {}
    for tag, records in raw.items():
        by_step = {}
        for step, wall, value in sorted(records, key=lambda record: record[1]):
            by_step[step] = value
        series[tag] = sorted(by_step.items())
    return series, issues


def window_mean(series, start, end):
    points = [(x, y) for x, y in series if math.isfinite(y)]
    if not points or points[-1][0] < end:
        return None
    xs, ys = map(np.asarray, zip(*points))
    interior = (xs > start) & (xs < end)
    x = np.concatenate(([start], xs[interior], [end]))
    y = np.concatenate(([np.interp(start, xs, ys)], ys[interior], [np.interp(end, xs, ys)]))
    return float(np.trapz(y, x) / (end - start))


configs = []
for path in ROOT.glob("sacred/*/*/*/config.json"):
    config = json.loads(path.read_text(encoding="utf-8"))
    run = json.loads((path.parent / "run.json").read_text(encoding="utf-8"))
    timestamp = (datetime.fromisoformat(run["start_time"]) + timedelta(hours=8)).strftime("%Y-%m-%d_%H-%M-%S")
    configs.append((config, run, path, timestamp))

rows = []
details = []
issues = []
for directory in sorted((ROOT / "tb_logs").iterdir()):
    if not directory.is_dir():
        continue
    name, timestamp = directory.name.rsplit("__", 1)
    if args.name_regex and not re.search(args.name_regex, name):
        continue
    matches = [item for item in configs if item[0].get("name") == name and item[3] == timestamp]
    if not matches:
        # Sacred captures the start just before the TB directory is created.
        matches = [item for item in configs if item[0].get("name") == name and abs((datetime.strptime(item[3], "%Y-%m-%d_%H-%M-%S") - datetime.strptime(timestamp, "%Y-%m-%d_%H-%M-%S")).total_seconds()) <= 2]
        if len(matches) == 1:
            issues.append({"directory": str(directory), "note": "matched unique same-name config within 2 seconds", "config_timestamp": matches[0][3]})
    if len(matches) != 1:
        issues.append({"directory": str(directory), "config_matches": len(matches)})
        continue
    config, run, path, _ = matches[0]
    series, event_issues = read_events(directory)
    issues.extend(event_issues)
    wins = series.get("test_battle_won_mean", [])
    method = re.sub(r"_s\d+$", "", name)
    if method == name:
        method = re.sub(r"_\d+(?=(_|$))", "", name)
    row = {
        "map": config["env_args"]["map_name"], "method": method,
        "name": name, "seed": config["seed"], "timestamp": timestamp,
        "run_id": path.parent.name, "status": run.get("status"),
        "mixer": config.get("mixer"), "learner": config.get("learner"),
        "epsilon_start": config.get("epsilon_start"),
        "epsilon_anneal_time": config.get("epsilon_anneal_time"),
        "n_eval": len(wins), "first_step": wins[0][0] if wins else None,
        "last_step": wins[-1][0] if wins else None,
        "final": wins[-1][1] if wins else None,
        "best": max((y for x, y in wins if math.isfinite(y)), default=None),
        "last5": float(np.mean([y for x, y in wins[-5:]])) if wins else None,
        "auc_0_10m": window_mean(wins, 0, 10_000_000),
        "full_10m": bool(wins and wins[-1][0] >= 10_000_000),
        "never_won": bool(wins and all(y == 0 for x, y in wins)),
        "nonfinite_tags": ";".join(tag for tag, values in series.items() if any(not math.isfinite(y) for x, y in values)),
        "config_path": str(path), "tb_path": str(directory),
    }
    for a, b in [(1, 3), (3, 5), (5, 7), (7, 9), (9, 10)]:
        row[f"win_{a}_{b}m"] = window_mean(wins, a * 1_000_000, b * 1_000_000)
    diagnostics = {tag: window_mean(values, 9_000_000, 10_000_000) for tag, values in series.items() if tag != "test_battle_won_mean"}
    rows.append(row)
    details.append({"row": row, "config": config, "run": run, "diagnostics_9_10m": diagnostics, "series": series})

with (OUT / "runs.csv").open("w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
with gzip.open(OUT / "details.json.gz", "wt", encoding="utf-8") as stream:
    json.dump(details, stream, ensure_ascii=False, separators=(",", ":"))
(OUT / "issues.json").write_text(json.dumps(issues, ensure_ascii=False, indent=2), encoding="utf-8")

groups = defaultdict(list)
for row in rows:
    if row["full_10m"] and not row["nonfinite_tags"]:
        groups[(row["map"], row["method"])].append(row)
summary = []
for (map_name, method), members in sorted(groups.items()):
    result = {"map": map_name, "method": method, "n": len(members), "seeds": [r["seed"] for r in members], "zero_runs": sum(r["never_won"] for r in members)}
    for metric in ["auc_0_10m", "win_1_3m", "win_3_5m", "win_7_9m", "win_9_10m", "last5", "final"]:
        values = [r[metric] for r in members]
        result[metric] = float(np.mean(values))
        result[metric + "_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
    summary.append(result)
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"configs": len(configs), "parsed_tb_runs": len(rows), "full_10m": sum(r["full_10m"] for r in rows), "issues": issues, "maps": dict(Counter(r["map"] for r in rows))}, ensure_ascii=True))
for result in summary:
    print(json.dumps({key: result[key] for key in ["map", "method", "n", "seeds", "zero_runs", "auc_0_10m", "win_9_10m", "last5"]}))
print('INCOMPLETE', json.dumps([{k:r[k] for k in ('name','last_step','never_won')} for r in rows if not r['full_10m']], ensure_ascii=True))
print('NUMERICALLY_INVALID', json.dumps([r['name'] + '__' + r['timestamp'] for r in rows if r['nonfinite_tags']]))
