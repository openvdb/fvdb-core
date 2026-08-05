# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Shared machinery for fetching auxiliary data repositories (example data, test data).

The example data lives in the shipped ``fvdb.utils.examples`` module and the test data
lives in ``fvdb.utils.tests``, so the fetch logic they both need lives here rather than
in either one.

We only ever need a read-only snapshot of a repository at a pinned revision, so this
downloads GitHub's source tarball with the standard library rather than depending on
git or GitPython. That also avoids transferring the repository history: the test data
tarball is ~129MB against ~659MB for a full clone.
"""

import logging
import os
import re
import shutil
import site
import tarfile
import tempfile
import urllib.request
from collections.abc import Iterator
from pathlib import Path

__all__ = ["fetch_data_repo", "local_repo_path"]

logger = logging.getLogger(__name__)

# fvdb/utils/_data_repo.py -> fvdb/utils -> fvdb -> <repo root>
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Name of the marker file we drop in an extracted snapshot to record which revision it holds.
_REVISION_MARKER = ".fvdb-data-revision"

_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

# Snapshot directory names must be a single, ordinary path component. We delete directories at
# these paths, and '', '.', and '..' would all resolve to the parent directory (which holds
# unrelated data) rather than to a snapshot of our own.
_REPO_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]*$")

_DOWNLOAD_TIMEOUT_SECONDS = 600

# GitHub's REST API requires a User-Agent. urllib sends "Python-urllib/<ver>" by default, which
# GitHub accepts, but an explicit one identifies the client and is not subject to that default
# changing underneath us.
_GITHUB_API_HEADERS = {
    "Accept": "application/vnd.github.sha",
    "User-Agent": "fvdb-core-data-fetcher",
}


def _is_editable_install() -> bool:
    # check we're not in a site package
    module_path = Path(__file__).resolve()
    for site_path in site.getsitepackages():
        if str(module_path).startswith(site_path):
            return False
    # check if we're in the source directory
    return (_REPO_ROOT / "pyproject.toml").is_file()


def local_repo_path(repo_name: str) -> Path:
    """Get the local path where a data repository snapshot should be unpacked.

    Args:
        repo_name: The name of the repository (e.g., 'fvdb_example_data', 'fvdb_test_data')

    Returns:
        Path to the local snapshot directory
    """
    if not _REPO_NAME_RE.match(repo_name):
        raise ValueError(f"Invalid snapshot directory name '{repo_name}'; expected a single path component")

    if _is_editable_install():
        external_dir = _REPO_ROOT / "external"
        external_dir.mkdir(exist_ok=True)
        base_path = external_dir
    else:
        base_path = Path(tempfile.gettempdir())

    return base_path / repo_name


def _resolve_revision(github_repo: str, revision: str) -> str:
    """Resolve a branch or tag name to the commit SHA it currently points at.

    Full SHAs are returned unchanged so the common (pinned) case needs no network access.
    """
    if _FULL_SHA_RE.match(revision):
        return revision

    # Branch and tag names are mutable, so resolve them to a SHA in order to tell a stale
    # snapshot from a current one. The .sha media type makes the response the bare SHA.
    url = f"https://api.github.com/repos/{github_repo}/commits/{revision}"
    request = urllib.request.Request(url, headers=_GITHUB_API_HEADERS)
    with urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
        resolved = response.read().decode("utf-8").strip()

    if not _FULL_SHA_RE.match(resolved):
        raise ValueError(f"Could not resolve revision '{revision}' of {github_repo}, got '{resolved}'")

    logger.info("Resolved %s@%s to %s", github_repo, revision, resolved)
    return resolved


def _current_revision(repo_path: Path) -> str | None:
    """Return the revision recorded in an existing snapshot, or None if there isn't one."""
    marker = repo_path / _REVISION_MARKER
    try:
        return marker.read_text().strip()
    except OSError:
        return None


def _checked_members(tar: tarfile.TarFile, dest_path: Path) -> Iterator[tarfile.TarInfo]:
    """Yield archive members, rejecting any that are unsafe to extract.

    Used only on interpreters that predate tarfile's extraction filters. It applies the checks
    that matter for the member types a GitHub source tarball contains: every member must be a
    regular file or a directory, and must land inside ``dest_path``. Symlinks, hard links,
    device nodes, absolute paths and paths containing ``..`` are rejected. A bad member raises
    rather than being skipped, so a tampered archive fails loudly instead of quietly extracting
    less than expected.
    """
    dest_root = dest_path.resolve()
    for member in tar.getmembers():
        if not (member.isfile() or member.isdir()):
            raise ValueError(
                f"Refusing to extract archive member {member.name!r}: expected a regular file or "
                f"directory, got type {member.type!r}"
            )
        target = (dest_root / member.name).resolve()
        if target != dest_root and dest_root not in target.parents:
            raise ValueError(f"Refusing to extract archive member {member.name!r} outside of {dest_root}")
        # Normalize permissions the way the "data" filter does, rather than trusting the modes
        # recorded in the archive. This drops setuid/setgid bits and guarantees the extracted
        # tree stays readable/traversable for the move into place below.
        member.mode = 0o755 if member.isdir() else 0o644
        yield member


def _extract_snapshot(archive_path: Path, dest_path: Path, sha: str) -> None:
    """Extract a GitHub source tarball, stripping its single top-level directory."""
    with tarfile.open(archive_path, mode="r:gz") as tar:
        # GitHub tarballs wrap everything in a '<repo>-<sha>/' directory that we strip off.
        try:
            tar.extractall(dest_path, filter="data")
        except TypeError:
            # The 'filter' argument only exists from Python 3.10.12 / 3.11.4 onwards, and this
            # project supports >=3.10, so this path is reachable. Validate the members by hand
            # rather than extracting unchecked.
            tar.extractall(dest_path, members=_checked_members(tar, dest_path))  # nosec B202

    extracted = [child for child in dest_path.iterdir() if child.is_dir()]
    if len(extracted) != 1:
        raise ValueError(f"Expected exactly one top-level directory in the archive, found {len(extracted)}")

    top_level = extracted[0]
    for child in top_level.iterdir():
        shutil.move(str(child), str(dest_path / child.name))
    top_level.rmdir()

    (dest_path / _REVISION_MARKER).write_text(sha + "\n")


def fetch_data_repo(github_repo: str, revision: str, repo_name: str) -> Path:
    """Download a snapshot of a public GitHub repository at a given revision.

    The snapshot is cached on disk and only re-downloaded when the requested revision
    differs from the one already present, so repeated calls are free.

    Args:
        github_repo: The repository in 'owner/name' form (e.g. 'voxel-foundation/fvdb-test-data')
        revision: A commit SHA, tag, or branch name to download
        repo_name: Name for the local snapshot directory

    Returns:
        Path to the downloaded snapshot
    """
    repo_path = local_repo_path(repo_name)
    sha = _resolve_revision(github_repo, revision)

    current = _current_revision(repo_path)
    if current == sha:
        logger.debug("Reusing %s snapshot at %s", repo_name, repo_path)
        return repo_path

    # We replace the destination below, which means deleting it. Only ever delete a directory we
    # recognise as one of our own snapshots: anything else at this path is somebody's data, not
    # ours. Check before downloading rather than after, so a refusal costs nothing.
    if current is None and repo_path.exists():
        raise ValueError(
            f"A path {repo_path} exists but is not an fvdb data snapshot "
            f"(no {_REVISION_MARKER} file). Refusing to delete it; "
            f"move or remove it by hand if it is no longer needed."
        )

    url = f"https://codeload.github.com/{github_repo}/tar.gz/{sha}"
    logger.info("Downloading %s@%s to %s", github_repo, sha, repo_path)

    # Stage the download and extraction alongside the destination so that an interrupted or
    # concurrent run can never leave a half-written snapshot in place. mkdtemp gives us a
    # uniquely-named directory that is guaranteed not to already exist, so the cleanup below
    # can only ever remove a directory this call created.
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = Path(tempfile.mkdtemp(prefix=f"{repo_path.name}.tmp-", dir=repo_path.parent))

    try:
        archive_path = staging_path / "archive.tar.gz"
        with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
            with open(archive_path, "wb") as archive_file:
                shutil.copyfileobj(response, archive_file)

        extract_path = staging_path / "extracted"
        extract_path.mkdir()
        _extract_snapshot(archive_path, extract_path, sha)
        archive_path.unlink()

        # os.replace needs the destination gone. Re-check the marker rather than trusting the
        # check above, in case a concurrent run put something else here while we downloaded.
        if repo_path.exists():
            if _current_revision(repo_path) is None:
                raise ValueError(f"A path {repo_path} appeared during download and is not an fvdb data snapshot")
            shutil.rmtree(repo_path)
        os.replace(extract_path, repo_path)
    finally:
        shutil.rmtree(staging_path, ignore_errors=True)

    return repo_path
