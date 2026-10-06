"""Resolve one output directory for Sacred, TensorBoard and model files."""
import json
from pathlib import Path


DEFAULT_RESULTS_PATH = "ablation_results_10_6"


def resolve_results_path(results_path, project_root=None):
    """Anchor relative paths to the project root; preserve absolute overrides."""
    if not isinstance(results_path, str) or not results_path.strip():
        raise ValueError("local_results_path must be a non-empty path string")
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    path = Path(results_path).expanduser()
    if not path.is_absolute():
        path = Path(project_root) / path
    return str(path.resolve())


def prepare_results_path(config, params, project_root=None):
    """Record the resolved path and normalize its Sacred CLI overrides.

    Sacred applies CLI overrides after add_config. Rewrite any path override
    to the same resolved value so the observer and training cannot diverge.
    """
    params = list(params)
    results_path = config.get("local_results_path", DEFAULT_RESULTS_PATH)
    positions = []
    for index, argument in enumerate(params):
        if "=" not in argument:
            continue
        key, value = argument.split("=", 1)
        if key.strip() == "local_results_path":
            import yaml
            results_path = yaml.safe_load(value)
            positions.append(index)
    resolved = resolve_results_path(results_path, project_root)
    config["local_results_path"] = resolved
    for index in positions:
        params[index] = "local_results_path=" + json.dumps(resolved, ensure_ascii=False)
    return params
