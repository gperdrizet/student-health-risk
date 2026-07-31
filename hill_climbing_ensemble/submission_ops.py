"""Submission generation and git automation for improved hill-climb checkpoints."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, UTC
from pathlib import Path
import re
import subprocess


ALLOWED_LABELS = {"unhealthy", "at-risk", "fit"}
EXPECTED_ROWS = 295753
EXPECTED_COLUMNS = ["id", "health_condition"]
EXPECTED_FIRST_ID = 690088
EXPECTED_LAST_ID = 985840
REPO_ROOT = Path(__file__).resolve().parents[1]


class GitAutomationError(RuntimeError):
    pass


SEMVER_TAG_PATTERN = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


def validate_submission_df(df) -> None:
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Submission row count mismatch: expected {EXPECTED_ROWS}, got {len(df)}")

    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Submission column mismatch: expected {EXPECTED_COLUMNS}, got {list(df.columns)}")

    if not set(df["health_condition"].dropna().unique()).issubset(ALLOWED_LABELS):
        raise ValueError("Submission contains invalid class values.")

    if int(df["id"].iloc[0]) != EXPECTED_FIRST_ID or int(df["id"].iloc[-1]) != EXPECTED_LAST_ID:
        raise ValueError("Submission boundary IDs do not match expected Kaggle format.")


def save_submission_csv(df, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def _run_git(args: list[str], dry_run: bool) -> str:
    command = ["git", *args]
    if dry_run:
        return f"DRY_RUN: {' '.join(command)}"

    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        raise GitAutomationError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def _next_semver_tag() -> str:
    result = subprocess.run(
        ["git", "tag", "--list"],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        raise GitAutomationError(result.stderr.strip() or result.stdout.strip())

    versions = []
    for raw_tag in result.stdout.splitlines():
        tag = raw_tag.strip()
        match = SEMVER_TAG_PATTERN.match(tag)
        if not match:
            continue
        major, minor, patch = map(int, match.groups())
        versions.append((major, minor, patch))

    if not versions:
        return "v0.1.0"

    major, minor, patch = max(versions)
    return f"v{major}.{minor}.{patch + 1}"


def _try_get_upstream_branch(dry_run: bool) -> str | None:
    if dry_run:
        return "origin/main"

    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        return None

    upstream = result.stdout.strip()
    return upstream or None


def _sync_with_upstream(dry_run: bool) -> None:
    _run_git(["fetch", "origin", "--tags"], dry_run=dry_run)
    upstream = _try_get_upstream_branch(dry_run=dry_run)
    if upstream:
        _run_git(["rebase", upstream], dry_run=dry_run)


@contextmanager
def _git_lock(lock_file: Path):
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = lock_file.open("x")
    except FileExistsError as exc:
        raise GitAutomationError(f"Git automation lock exists: {lock_file}") from exc

    try:
        fd.write(str(datetime.now(UTC)))
        fd.flush()
        yield
    finally:
        fd.close()
        lock_file.unlink(missing_ok=True)


def auto_commit_tag_push(
    *,
    commit_paths: tuple[str, ...],
    lock_file: Path,
    accepted_count: int,
    median_score: float,
    proposal_index: int,
    dry_run: bool = False,
):
    message = (
        f"model: hill-climb accepted model {accepted_count} "
        f"(median BA={median_score:.5f})"
    )

    with _git_lock(lock_file):
        # Keep branch aligned with upstream before staging and committing.
        _sync_with_upstream(dry_run=dry_run)

        _run_git(["add", *commit_paths], dry_run=dry_run)

        # Create a commit only if any staged changes exist.
        staged = _run_git(["diff", "--cached", "--name-only"], dry_run=dry_run)
        if dry_run:
            staged_has_changes = True
        else:
            staged_has_changes = bool(staged.strip())

        if staged_has_changes:
            _run_git(["commit", "-m", message], dry_run=dry_run)

        # Re-sync to absorb any remote updates from concurrent CI badge commits.
        _sync_with_upstream(dry_run=dry_run)

        _run_git(["push"], dry_run=dry_run)

        # Compute the next tag from freshly fetched tags to avoid stale increments.
        _run_git(["fetch", "origin", "--tags"], dry_run=dry_run)
        tag = _next_semver_tag()
        _run_git(["tag", tag], dry_run=dry_run)
        _run_git(["push", "origin", tag], dry_run=dry_run)

    return tag
