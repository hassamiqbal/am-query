"""Run in the existing notebook with: %run -i Cell_16_Storage_Fix.py"""
import dataclasses
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import sys

if "CFG" not in globals() or "Budget" not in globals():
    raise RuntimeError("Run your notebook's definition cells through Cell 15 first.")

def storage_paths(cfg):
    root = Path(cfg.root).expanduser().resolve()
    return {"run": root, "model_cache": (root.parent / "model_cache").resolve()}

def existing_directory(path):
    path = Path(path).expanduser().resolve()
    while not path.exists():
        path = path.parent
    if not path.is_dir():
        raise RuntimeError(f"Storage path is a file, not a directory: {path}")
    return path

def storage_report(cfg):
    report = {}
    for name, path in storage_paths(cfg).items():
        ancestor = existing_directory(path)
        report[name] = {"path": str(path), "checked_directory": str(ancestor),
                        "free_gib": shutil.disk_usage(ancestor).free / 2**30,
                        "writable": os.access(ancestor, os.W_OK | os.X_OK)}
    return report

def require_storage(cfg, create=False):
    report = storage_report(cfg)
    for name, entry in report.items():
        if not entry["writable"]:
            raise RuntimeError(f"{name} storage is not writable: {entry['path']}")
        if entry["free_gib"] < cfg.min_free_gb:
            raise RuntimeError(
                f"{name} storage has {entry['free_gib']:.2f} GiB free at {entry['path']}; "
                f"need at least {cfg.min_free_gb:.1f} GiB. Select assigned storage in "
                "Cell 16 (STORAGE_BASE), or set --root on the command line. "
                "Changing HF_HOME alone does not move this project's explicit model cache. "
                "Do not lower the threshold to bypass a full filesystem.")
    if create:
        import tempfile
        for entry in report.values():
            directory = Path(entry["path"])
            directory.mkdir(parents=True, exist_ok=True)
            # This detects write/permission failures, not remaining per-user quota.
            with tempfile.TemporaryFile(dir=directory) as probe:
                probe.write(b"AM-Query storage check\n")
                probe.flush()
                os.fsync(probe.fileno())
    return report

def configure_storage(cfg, storage_base=""):
    """Choose assigned storage for a NEW run; never move or delete existing data.

    An explicit base is a directory the operator is authorised to use. Automatic
    candidates are existing, user-owned directories from allocation variables.
    No temporary-directory variable or arbitrary shared mount is selected.
    """
    original = Path(cfg.root).expanduser().resolve()
    if not storage_base:
        try:
            require_storage(cfg)
            cfg.root = str(original)
            return cfg.root
        except (OSError, RuntimeError):
            pass
    candidates = []
    if storage_base:
        candidates.append(("STORAGE_BASE", Path(storage_base).expanduser().resolve()))
    else:
        for name in ("AM_QUERY_STORAGE", "SCRATCH", "WORK", "PROJECT"):
            value = os.environ.get(name)
            if value:
                path = Path(value).expanduser().resolve()
                try:
                    owned = hasattr(os, "getuid") and path.stat().st_uid == os.getuid()
                    if path.is_dir() and owned:
                        candidates.append((name, path))
                except OSError:
                    pass
    failures = []
    for name, base in candidates:
        try:
            if not base.is_dir():
                raise RuntimeError("Select an existing assigned directory")
            root = (base / "am_query_runs" / cfg.profile).resolve()
            candidate = dataclasses.replace(cfg, root=str(root))
            require_storage(candidate)
            if original != root and original.exists() and any(original.iterdir()):
                raise RuntimeError(
                    "Existing run files found. Preserve the original run: copy its complete "
                    "folder to assigned storage and explicitly set CFG.root to the copied "
                    "folder before continuing. Nothing has been moved or deleted.")
            cfg.root = str(root)
            print(f"Selected {name}: {base}")
            print("Confirm this directory's retention period and your individual storage quota.")
            return cfg.root
        except (OSError, RuntimeError) as exc:
            failures.append(f"{name}: {exc}")
    details = "\n".join(failures)
    raise RuntimeError(
        f"No suitable assigned storage found with {cfg.min_free_gb:.1f} GiB free. "
        "Run storage_diagnostics(CFG), then set STORAGE_BASE to your existing assigned "
        "project/scratch directory. If none has enough capacity, request more storage. "
        "No files were deleted and no download started.\n" + details)

def storage_diagnostics(cfg):
    import subprocess
    print(json.dumps(storage_report(cfg), indent=2))
    for name in ("AM_QUERY_STORAGE", "SCRATCH", "WORK", "PROJECT", "LOCAL_SCRATCH", "SLURM_TMPDIR"):
        value = os.environ.get(name)
        if value:
            print(f"{name}={value}")
    try:
        result = subprocess.run(["df", "-hP"], capture_output=True, text=True, timeout=10)
        print(result.stdout or result.stderr)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"Could not list mounted filesystems: {exc}")
    print("Filesystem free space is not a measurement of your personal/project quota.")
    print("Temporary allocation storage may disappear when the allocation ends.")

def preflight(cfg):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Select the allocated NVIDIA GPU Jupyter kernel first.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This configuration requires a GPU with BF16 support.")
    if sys.version_info < (3,10):
        raise RuntimeError("Use Python 3.10, 3.11 or 3.12.")
    mismatches = {}
    for package, expected in DEPENDENCIES.items():
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            actual = "missing"
        if actual != expected:
            mismatches[package] = {"installed":actual, "expected":expected}
    if mismatches:
        raise RuntimeError("Run the notebook setup cell in a dedicated kernel, then restart: " + json.dumps(mismatches))
    root = Path(cfg.root).expanduser().resolve()
    cfg.root = str(root)
    storage = storage_report(cfg)
    disk_gb = storage["run"]["free_gib"]
    free, total = torch.cuda.mem_get_info(0)
    result = {"torch":torch.__version__, "cuda":torch.version.cuda,
              "gpu":torch.cuda.get_device_name(0), "gpu_free_gib":round(free/2**30,2),
              "gpu_total_gib":round(total/2**30,2), "disk_free_gib":round(disk_gb,2),
              "storage":storage, "python":sys.version.split()[0], "packages":DEPENDENCIES,
              "deadline":cfg.deadline, "session_hours":cfg.session_hours}
    print(json.dumps(result, indent=2))
    if free < 14*2**30:
        raise RuntimeError("Less than 14 GiB GPU memory is currently free; check the allocation.")
    require_storage(cfg, create=True)
    Budget(cfg).check()
    return result

try:
    configure_storage(CFG, storage_base=globals().get("STORAGE_BASE", ""))
except RuntimeError:
    storage_diagnostics(CFG)
    raise
GPU_REPORT = preflight(CFG)
