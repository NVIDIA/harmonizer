#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate host prerequisites for the DiffusionHarmonizer skill.

Checks performed without network calls:
  1. docker is on PATH.
  2. NVIDIA Container Toolkit is available.
  3. nvidia-smi reports at least one GPU with compute capability >= 8.0.
  4. git is on PATH for cloning https://github.com/NVIDIA/harmonizer.
  5. Hugging Face CLI is available.
  6. HF_TOKEN is set without echoing its value.
  7. Optional NGC_API_KEY presence is reported for nvcr.io pulls.
  8. At least 120 GB of free disk space is available in the cwd.

Usage:
    python scripts/validate_setup.py [--strict]

Arguments:
    --strict    Treat token/tool warnings as errors.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

MIN_COMPUTE_CAPABILITY = (8, 0)
MIN_FREE_DISK_GB = 120
SUBPROCESS_TIMEOUT_S = 30
BYTES_PER_GB = 1024**3
UNEXPECTED_ERROR_EXIT = 2


def _err(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def _ok(msg: str) -> None:
    print(f"[ OK ] {msg}")


def check_executable(name: str, install_hint: str, strict: bool = True) -> bool:
    if shutil.which(name) is not None:
        _ok(f"{name} found")
        return True
    if strict:
        _err(f"{name} not on PATH - {install_hint}")
        return False
    _warn(f"{name} not on PATH - {install_hint}")
    return True


def check_docker() -> bool:
    return check_executable("docker", "install Docker from https://docs.docker.com/engine/install/")


def check_nvidia_runtime() -> bool:
    if shutil.which("nvidia-container-cli") is not None:
        _ok("nvidia-container-cli found")
        return True
    try:
        out = subprocess.run(
            ["docker", "info"],
            check=False,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_S,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        _err(f"could not run `docker info`: {exc}")
        return False
    if "nvidia" in (out.stdout or "").lower():
        _ok("nvidia runtime reported by `docker info`")
        return True
    _err(
        "NVIDIA Container Toolkit not detected - install from "
        "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
    )
    return False


def check_gpu() -> bool:
    if shutil.which("nvidia-smi") is None:
        _err("nvidia-smi not on PATH - is an NVIDIA driver installed?")
        return False
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_S,
        )
    except subprocess.CalledProcessError as exc:
        _err(f"nvidia-smi failed: {exc.stderr.strip() or exc}")
        return False
    except subprocess.TimeoutExpired:
        _err("nvidia-smi timed out")
        return False

    lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    if not lines:
        _err("nvidia-smi reported no GPUs")
        return False

    bad: list[str] = []
    for line in lines:
        name, _, cc_str = line.partition(",")
        cc_str = cc_str.strip()
        try:
            major_s, minor_s = cc_str.split(".")
            cc = (int(major_s), int(minor_s))
        except ValueError:
            _err(f"could not parse compute capability from: {line!r}")
            return False
        if cc < MIN_COMPUTE_CAPABILITY:
            bad.append(f"{name.strip()} (sm_{cc[0]}{cc[1]})")
        else:
            _ok(f"{name.strip()} sm_{cc[0]}{cc[1]} meets minimum sm_80")
    if bad:
        _err("GPU(s) below minimum compute capability 8.0 (Ampere): " + ", ".join(bad))
        return False
    return True


def check_hf_cli(strict: bool) -> bool:
    if shutil.which("hf") is not None:
        _ok("hf CLI found")
        return True
    if shutil.which("huggingface-cli") is not None:
        _ok("huggingface-cli found")
        return True
    msg = 'Hugging Face CLI not on PATH - install with `python3 -m pip install --user "huggingface_hub[cli]"`'
    if strict:
        _err(msg)
        return False
    _warn(msg)
    return True


# Placeholders shipped in .env.example. These are not credentials, and an
# exported sentinel overrides a real cached login and makes downloads fail.
_TOKEN_SENTINELS = frozenset(
    {
        "hf_replace_me",
        "hf_your_token_here",
        "nvapi-replace-me",
        "replace_me",
        "changeme",
    }
)


def _hf_cached_token_path() -> str | None:
    """Effective cached `hf auth login` path, using huggingface_hub precedence.

    HF_TOKEN_PATH wins outright, else HF_HOME/token, else
    XDG_CACHE_HOME/huggingface/token, else the default. Only the winning path
    is consulted, and a whitespace-only file is not a credential.
    """

    def _expand(p: str) -> str:
        return os.path.expanduser(os.path.expandvars(p))

    explicit = os.environ.get("HF_TOKEN_PATH")
    if explicit:
        path = _expand(explicit)
    elif os.environ.get("HF_HOME"):
        path = os.path.join(_expand(os.environ["HF_HOME"]), "token")
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        base = _expand(xdg) if xdg else os.path.join(os.path.expanduser("~"), ".cache")
        path = os.path.join(base, "huggingface", "token")
    try:
        with open(path, encoding="utf-8") as handle:
            if handle.read().strip():
                return path
    except OSError:
        return None
    return None


def check_env_token(name: str, purpose: str, strict: bool) -> bool:
    value = (os.environ.get(name) or "").strip()
    if value and value in _TOKEN_SENTINELS:
        msg = (
            f"{name} is set to the placeholder '{value}'. Replace it with a real "
            "value or unset it - as exported it is treated as a real credential "
            "and will be used in place of any working one."
        )
        # Only fail where the credential is actually required. An optional
        # placeholder must not hard-fail the default run.
        if strict:
            _err(msg)
            return False
        _warn(msg)
        return True
    if value:
        _ok(f"{name} is set ({len(value)} chars)")
        return True
    if name == "HF_TOKEN":
        cached = _hf_cached_token_path()
        if cached:
            _ok(f"cached Hugging Face login found ({cached})")
            return True
    msg = f"{name} is not set - {purpose}"
    if strict:
        _err(msg)
        return False
    _warn(msg)
    return True


def check_disk_space() -> bool:
    cwd = Path.cwd()
    try:
        free_bytes = shutil.disk_usage(cwd).free
    except OSError as exc:
        _err(f"could not check free disk: {exc}")
        return False
    free_gb = free_bytes / BYTES_PER_GB
    if free_gb < MIN_FREE_DISK_GB:
        _err(f"only {free_gb:.1f} GB free in {cwd}; recommend >= {MIN_FREE_DISK_GB} GB")
        return False
    _ok(f"{free_gb:.1f} GB free in {cwd}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors.")
    args = parser.parse_args()

    checks = [
        ("docker", check_docker()),
        ("nvidia-container-toolkit", check_nvidia_runtime()),
        ("gpu", check_gpu()),
        ("git", check_executable("git", "install git", strict=True)),
        ("hf-cli", check_hf_cli(args.strict)),
        (
            "hf-token",
            check_env_token(
                "HF_TOKEN",
                "needed for the gated base Cosmos model and, for training, the dataset - "
                "nvidia/Harmonizer itself is public and needs no token",
                args.strict,
            ),
        ),
        (
            "ngc-api-key",
            check_env_token(
                "NGC_API_KEY",
                "often required for `docker login nvcr.io` before pulling NVIDIA containers",
                strict=False,
            ),
        ),
        ("disk-space", check_disk_space()),
    ]
    failed = [name for name, ok in checks if not ok]
    if failed:
        print(f"\nFAILED: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("\nPrerequisite check completed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[ERROR] unexpected failure: {exc}", file=sys.stderr)
        raise SystemExit(UNEXPECTED_ERROR_EXIT) from exc
