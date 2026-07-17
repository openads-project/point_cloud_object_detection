#!/usr/bin/env python3

# Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
# SPDX-License-Identifier: Apache-2.0

"""Prepare writable demo data before starting the autonomy_datasets image.

The upstream image runs as root and writes generated datasets into a host bind
mount. This wrapper fixes ownership so the host user can delete those files
again, and removes rosbag directories left incomplete by interrupted demo runs.
"""

import os
import pwd
import shutil
import sys
from pathlib import Path

DEFAULT_UID = 1000
DEFAULT_GID = 1000
DATASET_DIR = Path("/datasets")
DATA_PARENT_DIR = Path("/demo-data")
OWNER_REFERENCE = Path("/config/autonomy_datasets.params.yml")


def _configured_id(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as error:
        raise SystemExit(f"{name} must be numeric, got {value!r}") from error


def _target_ids(path: Path) -> tuple[int, int]:
    stat_result = path.stat()
    reference_stat_result = OWNER_REFERENCE.stat() if OWNER_REFERENCE.exists() else None
    uid = _configured_id("HOST_UID")
    gid = _configured_id("HOST_GID")

    target_uid = uid if uid is not None else stat_result.st_uid
    target_gid = gid if gid is not None else stat_result.st_gid

    if target_uid == 0 and reference_stat_result is not None:
        target_uid = reference_stat_result.st_uid
    if target_gid == 0 and reference_stat_result is not None:
        target_gid = reference_stat_result.st_gid

    if target_uid == 0:
        target_uid = DEFAULT_UID
    if target_gid == 0:
        target_gid = DEFAULT_GID

    return target_uid, target_gid


def _chown_tree(path: Path, uid: int, gid: int) -> None:
    _chown_if_needed(path, uid, gid)
    for root, dirs, files in os.walk(path):
        for directory in dirs:
            _chown_if_needed(Path(root, directory), uid, gid)
        for file in files:
            _chown_if_needed(Path(root, file), uid, gid)


def _chown_if_needed(path: Path, uid: int, gid: int) -> None:
    stat_result = path.stat()
    if stat_result.st_uid != uid or stat_result.st_gid != gid:
        os.chown(path, uid, gid)


def _data_roots_have_owner(uid: int, gid: int) -> bool:
    return _has_owner(DATA_PARENT_DIR, uid, gid) and _has_owner(DATASET_DIR, uid, gid)


def _has_owner(path: Path, uid: int, gid: int) -> bool:
    stat_result = path.stat()
    return stat_result.st_uid == uid and stat_result.st_gid == gid


def _drop_privileges(uid: int, gid: int) -> None:
    os.setgroups([])
    os.setgid(gid)
    os.setuid(uid)

    os.environ.setdefault("HOME", "/tmp")
    os.environ.setdefault("HF_HOME", str(DATASET_DIR / ".cache" / "huggingface"))
    os.environ.setdefault("XDG_CACHE_HOME", str(DATASET_DIR / ".cache"))

    try:
        user_name = pwd.getpwuid(uid).pw_name
    except KeyError:
        user_name = str(uid)
    os.environ.setdefault("USER", user_name)
    os.environ.setdefault("LOGNAME", user_name)


def _remove_incomplete_rosbags() -> None:
    for bags_dir in DATASET_DIR.glob("*/bags"):
        if not bags_dir.is_dir():
            continue

        for bag_dir in bags_dir.iterdir():
            if not bag_dir.is_dir():
                continue
            if (bag_dir / "metadata.yaml").is_file():
                continue

            print(f"Removing incomplete rosbag directory: {bag_dir}", flush=True)
            shutil.rmtree(bag_dir)


def main() -> None:
    """Prepare writable dataset directories and run the wrapped command."""
    if len(sys.argv) < 2:
        raise SystemExit("Usage: prepare_and_run.py COMMAND [ARGS...]")

    DATA_PARENT_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    uid, gid = _target_ids(DATA_PARENT_DIR)

    if os.getuid() == 0:
        print(f"Preparing demo data directory for uid={uid}, gid={gid}", flush=True)
        if _data_roots_have_owner(uid, gid):
            print("Demo data directory ownership is already correct", flush=True)
        else:
            print("Fixing demo data directory ownership", flush=True)
            _chown_tree(DATA_PARENT_DIR, uid, gid)
        _remove_incomplete_rosbags()
        _drop_privileges(uid, gid)

    print(f"Starting command: {' '.join(sys.argv[1:])}", flush=True)
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
