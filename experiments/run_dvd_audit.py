"""Print or execute a bounded experiment queue using only the new entry point.

Defaults to printing commands. Execution requires --run and explicit GPU ids.
Place this file in src/experiments when using the server's PyMARL layout.
"""
import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "legacy": (False, False),
    "td": (True, False),
    "aux": (False, True),
    "both": (True, True),
}


def command_for(method, variant, seed, map_name, t_max=None):
    td, aux = VARIANTS[variant]
    if method != "router" and aux:
        raise ValueError("aux/both variants apply only to the counterfactual router")
    name = "audit_{}_{}_{}_s{}".format(method, variant, map_name, seed)
    command = [sys.executable, str(ROOT / "main_audit.py"),
               "--config=dvd_audit_" + method, "--env-config=sc2", "with",
               "env_args.map_name=" + map_name, "use_tensorboard=True",
               "seed=" + str(seed), "name=" + name,
               "audit_fix_td_lambda=" + str(td),
               "audit_isolate_dvd_aux_hidden=" + str(aux)]
    if t_max is not None:
        command.append("t_max=" + str(t_max))
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("router", "base", "bm"), default="router")
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[79])
    parser.add_argument("--map", default="3s5z_vs_3s6z")
    parser.add_argument("--gpus", default=None, help="comma-separated physical GPU ids")
    parser.add_argument("--t-max", type=int, default=None)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    variants = args.variants or (list(VARIANTS) if args.method == "router" else ["legacy", "td"])
    jobs = [command_for(args.method, variant, seed, args.map, args.t_max)
            for seed in args.seeds for variant in variants]
    gpus = args.gpus.split(",") if args.gpus else []
    if len(set(gpus)) != len(gpus) or any(not gpu.isdigit() for gpu in gpus):
        parser.error("--gpus must contain unique non-negative integer ids")
    if args.run and not gpus:
        parser.error("--run requires explicit --gpus (for example 0,1,2,3)")
    if not args.run:
        for i, command in enumerate(jobs):
            prefix = "CUDA_VISIBLE_DEVICES=" + gpus[i % len(gpus)] + " " if gpus else ""
            print(prefix + " ".join(shlex.quote(part) for part in command))
        return

    active = {}
    failed = []
    try:
        while jobs or active:
            for gpu in gpus:
                if gpu not in active and jobs:
                    command = jobs.pop(0)
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = gpu
                    print("Launching on GPU {}: {}".format(gpu, shlex.join(command)), flush=True)
                    active[gpu] = (subprocess.Popen(command, cwd=str(ROOT), env=env), command)
            for gpu, (process, command) in list(active.items()):
                result = process.poll()
                if result is not None:
                    if result != 0:
                        failed.append((result, command))
                        # Do not spend the rest of the queue on a broken setup.
                        jobs.clear()
                    del active[gpu]
            if active:
                time.sleep(1)
    except KeyboardInterrupt:
        for process, command in active.values():
            process.terminate()
        for process, command in active.values():
            process.wait()
        raise
    if failed:
        for code, command in failed:
            print("FAILED ({}) {}".format(code, shlex.join(command)), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
