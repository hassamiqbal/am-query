"""Run in the existing notebook with: %run -i Cell_16_Storage_Fix.py"""
import contextlib
import dataclasses
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import sys

if "CFG" not in globals() or "Budget" not in globals():
    raise RuntimeError("Run your notebook's definition cells through Cell 15 first.")

@contextlib.contextmanager
def run_lock(root):
    # The OS releases this lock if the process exits or is killed.
    import errno
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("a+b") as handle:
        if sys.platform == "win32":
            import msvcrt
            # Windows supports locking a region beyond EOF: no byte needs writing.
            def set_lock(unlock=False):
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK if unlock else msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            def set_lock(unlock=False):
                fcntl.flock(handle, fcntl.LOCK_UN if unlock else fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            set_lock()
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise RuntimeError("This experiment is already running in another process.") from exc
            raise
        try:
            yield
        finally:
            set_lock(unlock=True)


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
                    if path.is_dir() and (owned or name == "AM_QUERY_STORAGE"):
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

def windows_drive_report():
    """List visible Windows drive capacity without selecting or writing to a drive."""
    import string
    drives = os.listdrives() if hasattr(os, "listdrives") else [f"{letter}:/" for letter in string.ascii_uppercase]
    report = []
    for drive in drives:
        try:
            usage = shutil.disk_usage(drive)
            report.append({"drive": drive, "free_gib": round(usage.free / 2**30, 2),
                           "total_gib": round(usage.total / 2**30, 2)})
        except OSError:
            # Disconnected mapped drives and empty removable-media drives are common.
            continue
    return report

def kernel_diagnostics():
    report = {"platform": sys.platform, "python_executable": sys.executable,
              "cuda_available": False}
    try:
        import torch
        report.update(torch=torch.__version__, cuda_available=torch.cuda.is_available())
        if report["cuda_available"]:
            report["gpu"] = torch.cuda.get_device_name(0)
            report["gpu_free_gib"] = round(torch.cuda.mem_get_info(0)[0] / 2**30, 2)
    except (ImportError, OSError, RuntimeError) as exc:
        report["gpu_check_error"] = str(exc)
    print(json.dumps(report, indent=2))
    return report

def storage_diagnostics(cfg):
    import subprocess
    print(json.dumps(storage_report(cfg), indent=2))
    for name in ("AM_QUERY_STORAGE", "SCRATCH", "WORK", "PROJECT", "LOCAL_SCRATCH", "SLURM_TMPDIR"):
        value = os.environ.get(name)
        if value:
            print(f"{name}={value}")
    if sys.platform == "win32":
        try:
            drives = windows_drive_report()
            print("Windows drives (capacity only; choose a directory you are authorised to use):")
            print(json.dumps(drives, indent=2))
            if not any(d["free_gib"] >= cfg.min_free_gb for d in drives):
                print(f"No listed drive has {cfg.min_free_gb:.1f} GiB free. Free known files, "
                      "request additional storage, or use your GPU server's assigned storage.")
        except OSError as exc:
            print(f"Could not list Windows drives: {exc}")
        print("Set STORAGE_BASE to an existing assigned folder, for example r'D:\\AM_Query' "
              "only if that folder exists on a suitable authorised drive.")
    else:
        try:
            result = subprocess.run(["df", "-hP"], capture_output=True, text=True, timeout=10)
            print(result.stdout or result.stderr)
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"Could not list mounted filesystems: {exc}")
    print("Filesystem free space is not a measurement of your personal/project quota.")
    print("Temporary allocation storage may disappear when the allocation ends.")

def notebook_preflight(cfg, storage_base=""):
    kernel = kernel_diagnostics()
    if not kernel["cuda_available"]:
        storage_diagnostics(cfg)
        raise RuntimeError(
            "This Jupyter kernel cannot access a CUDA GPU. Open this notebook on your "
            "allocated L40S server/kernel, then rerun the definition cells. A notebook "
            "on another computer cannot use that GPU just by changing a storage path.")
    try:
        configure_storage(cfg, storage_base=storage_base)
    except RuntimeError:
        storage_diagnostics(cfg)
        raise
    return preflight(cfg)

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
              "storage":storage, "platform":sys.platform, "python_executable":sys.executable,
              "python":sys.version.split()[0], "packages":DEPENDENCIES,
              "deadline":cfg.deadline, "session_hours":cfg.session_hours}
    print(json.dumps(result, indent=2))
    if free < 14*2**30:
        raise RuntimeError("Less than 14 GiB GPU memory is currently free; check the allocation.")
    require_storage(cfg, create=True)
    Budget(cfg).check()
    return result

GPU_REPORT = notebook_preflight(CFG, storage_base=globals().get("STORAGE_BASE", ""))
