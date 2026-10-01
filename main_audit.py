"""Optional experiment entry point; registers only the new audit learner.

The existing main.py, registries, learner, and configurations are untouched.
Use --audit-print-config to inspect flags without importing SMAC or Sacred.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUDIT_CONFIGS = ("dvd_audit_router", "dvd_audit_base", "dvd_audit_bm")
SOURCE_FILES = (
    "main_audit.py", "main.py", "learners/dvd_audit_learner.py",
    "utils/td_lambda_audit.py", "modules/mixers/dvd_counterfactual_router_audit.py",
    "modules/mixers/dvd_counterfactual_router.py",
    "modules/mixers/dvd_adaptive_takeover.py", "modules/mixers/dvd_takeover.py",
    "modules/mixers/dvd_credit.py", "modules/mixers/dvd_residual.py",
    "modules/mixers/dvd.py", "modules/mixers/nmix.py",
    "modules/agents/n_rnn_agent.py", "controllers/n_controller.py",
    "modules/exploration/rnd.py", "run/run.py", "components/episode_buffer.py",
)


def _merge(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def _read_yaml(path):
    import yaml
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def load_configuration(argv):
    params = list(argv)
    selected = {}
    for flag in ("--config", "--env-config"):
        matches = [arg for arg in params if arg.startswith(flag + "=")]
        if len(matches) != 1:
            raise ValueError("supply exactly one " + flag + "=NAME")
        selected[flag] = matches[0].split("=", 1)[1]
        params.remove(matches[0])
    if selected["--config"] not in AUDIT_CONFIGS:
        raise ValueError("audit entry expects one of: " + ", ".join(AUDIT_CONFIGS))
    env_path = ROOT / "config" / "envs" / (selected["--env-config"] + ".yaml")
    # Env names are filenames, not arbitrary paths outside config/envs.
    if env_path.resolve().parent != (ROOT / "config" / "envs").resolve():
        raise ValueError("invalid environment configuration name")
    alg_path = ROOT / "config" / "algs" / (selected["--config"] + ".yaml")
    config = _read_yaml(ROOT / "config" / "default.yaml")
    _merge(config, _read_yaml(env_path))
    _merge(config, _read_yaml(alg_path))
    sources = SOURCE_FILES + (
        "config/default.yaml", str(env_path.relative_to(ROOT)),
        str(alg_path.relative_to(ROOT)),
    )
    config["audit_source_manifest"] = [
        {"path": name.replace("\\", "/"),
         "sha256": hashlib.sha256((ROOT / name).read_bytes()).hexdigest()}
        for name in sources
    ]
    return config, params, sources


def inspect_overrides(config, params):
    """Preview ordinary Sacred key=value overrides; not a Sacred parser."""
    import yaml
    for arg in params:
        if "=" not in arg or arg.startswith("--"):
            continue
        key, value = arg.split("=", 1)
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = yaml.safe_load(value)
    return config


def main(argv=None):
    config, params, sources = load_configuration(sys.argv[1:] if argv is None else argv)
    if "--audit-print-config" in params:
        params.remove("--audit-print-config")
        print(json.dumps(inspect_overrides(config, params), indent=2, ensure_ascii=False))
        return

    # Imports are delayed so config inspection and tests need no SC2 install.
    import main as original_entry
    from learners import REGISTRY
    from learners.dvd_audit_learner import DVDNQLearner
    from sacred.observers import FileStorageObserver

    REGISTRY["dvd_audit_learner"] = DVDNQLearner
    ex = original_entry.ex
    for name in sources:
        ex.add_source_file(str(ROOT / name))
    ex.add_config(config)
    map_name = original_entry.parse_command(
        params, "env_args.map_name", config["env_args"]["map_name"]
    )
    algo_name = original_entry.parse_command(params, "name", config["name"])
    observer_path = Path(original_entry.results_path) / "sacred" / map_name / algo_name
    original_entry.logger.info("Audit experiment records: %s", observer_path)
    ex.observers.append(FileStorageObserver.create(str(observer_path)))
    ex.run_commandline([str(ROOT / "main_audit.py")] + params)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
