"""Provenance written into run manifests: the component commit and the pyoomph revision."""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
from pathlib import Path


def component_commit(root: Path) -> str:
    """Commit of the component source at ``root``.

    A source tree materialised from an archive carries its commit in a ``COMMIT`` file;
    otherwise git HEAD, marked ``-dirty`` when tracked files differ from it, and ``unknown``
    when neither is available.
    """
    commit_file = root / "COMMIT"
    if commit_file.is_file():
        return commit_file.read_text().strip()
    try:
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
                                         stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                                        text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + ("-dirty" if dirty else "")


def pyoomph_revision() -> dict[str, str | None]:
    """Version, source repository and commit of the installed pyoomph, from its installation record.

    pyoomph is pinned to a git revision (``pyproject.toml``, ``uv.lock``); the installer records
    that revision in the distribution's ``direct_url.json``. Nothing is guessed: without an
    installation record the version is ``None``, and without the git record the repository
    and commit are ``None``.
    """
    info: dict[str, str | None] = {"version": None, "url": None, "commit": None}
    try:
        dist = importlib.metadata.distribution("pyoomph")
    except importlib.metadata.PackageNotFoundError:
        return info
    info["version"] = dist.version
    record = dist.read_text("direct_url.json")
    if record:
        direct = json.loads(record)
        info["url"] = direct.get("url")
        info["commit"] = (direct.get("vcs_info") or {}).get("commit_id")
    return info
