"""Install the bounded direct-control payload after checking every target.

Distributed next to direct_ab_patch_manifest.json and the src/ payload.
Existing user edits are never overwritten without a known input hash.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def install(package, root, check_only=False):
    package, root = Path(package).resolve(), Path(root).resolve()
    if not (root / "src/main.py").is_file():
        raise ValueError("--root must be the pymarl2 directory containing src/main.py")
    manifest = json.loads((package / "direct_ab_patch_manifest.json").read_text(encoding="utf-8"))
    pending = []
    for item in manifest["files"]:
        relative = Path(item["path"])
        target, source = (root / relative).resolve(), (package / relative).resolve()
        if root not in target.parents or package not in source.parents:
            raise ValueError("payload path escapes its root: " + str(relative))
        data = source.read_bytes()
        if digest(data) != item["sha256"]:
            raise ValueError("payload hash mismatch: " + str(relative))
        old = target.read_bytes() if target.exists() else None
        current = digest(old) if old is not None else None
        if current == item["sha256"]:
            continue
        if current != item["previous_sha256"]:
            raise ValueError("unexpected existing version; no files changed: " + str(target))
        pending.append((relative, target, data, old))
    if check_only:
        return {"status": "checked", "pending_files": len(pending)}
    if not pending:
        return {"status": "already_installed", "changed_files": 0}

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup = root / "ablation_results_dvd/source_audit" / ("direct_ab_install_" + stamp)
    backup.mkdir(parents=True, exist_ok=False)
    # Complete backups before changing any target.
    for relative, target, data, old in pending:
        if old is not None:
            saved = backup / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_bytes(old)
    (backup / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for relative, target, data, old in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return {"status": "installed", "changed_files": len(pending), "backup": str(backup)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--check", action="store_true")
    options = parser.parse_args()
    result = install(Path(__file__).resolve().parent, options.root, options.check)
    print(json.dumps(result, indent=2))
