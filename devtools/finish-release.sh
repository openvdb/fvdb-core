#!/usr/bin/env bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# Finish a release: tag, merge PR, create GitHub Release.
#
# Usage: ./devtools/finish-release.sh <version> [options]
# Example: ./devtools/finish-release.sh 0.4.0 --remote upstream

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" \
    || { echo "error: must be run from inside a git repository" >&2; exit 1; }

# --- defaults ----------------------------------------------------------------
REMOTE="upstream"
DRY_RUN=false
NO_PUSH=false
NO_PR=false

# --- helpers ------------------------------------------------------------------
usage() {
    cat <<EOF
Usage: $(basename "$0") <version> [options]

Finish a release by tagging the release branch, merging the PR into main, and
creating a GitHub Release.

Arguments:
  <version>       Release version in MAJOR.MINOR.PATCH format (e.g. 0.4.0)

Options:
  --remote NAME   Git remote to push to (default: upstream)
  --dry-run       Print what would happen without making changes
  --no-push       Skip git push and GitHub operations (for testing)
  --no-pr         Skip PR merge and GitHub Release creation (for testing)
  -h, --help      Show this help message
EOF
}

die() { echo "error: $*" >&2; exit 1; }

log() { echo "==> $*"; }

run() {
    if $DRY_RUN; then
        echo "[dry-run] $*"
    else
        "$@"
    fi
}

release_branch_suffix() {
    local ver="$1"
    local major minor _patch
    IFS='.' read -r major minor _patch <<< "$ver"
    echo "${major}.${minor}"
}

# --- argument parsing ---------------------------------------------------------
VERSION=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --remote)   REMOTE="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=true; shift ;;
        --no-push)  NO_PUSH=true; shift ;;
        --no-pr)    NO_PR=true; shift ;;
        -h|--help)  usage; exit 0 ;;
        -*)         die "unknown option: $1" ;;
        *)
            [[ -z "$VERSION" ]] || die "unexpected argument: $1"
            VERSION="$1"; shift
            ;;
    esac
done

[[ -n "$VERSION" ]] || { usage; die "version argument is required"; }
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "version must be MAJOR.MINOR.PATCH (got: $VERSION)"

# --- derived values -----------------------------------------------------------
BRANCH_SUFFIX="$(release_branch_suffix "$VERSION")"
RELEASE_BRANCH="release/v${BRANCH_SUFFIX}"
TAG="v${VERSION}"

# --- pre-flight checks -------------------------------------------------------
cd "$REPO_ROOT"

log "Release version:     $VERSION"
log "Release branch:      $RELEASE_BRANCH"
log "Tag:                 $TAG"
log "Remote:              $REMOTE"
echo ""

if ! $DRY_RUN; then
    if ! git show-ref --verify --quiet "refs/heads/$RELEASE_BRANCH"; then
        die "branch $RELEASE_BRANCH does not exist"
    fi

    if git rev-parse "$TAG" >/dev/null 2>&1; then
        die "tag $TAG already exists"
    fi
fi

# --- verify publish workflow passed -------------------------------------------
if ! $DRY_RUN && ! $NO_PR; then
    REPO_SLUG="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
    log "Checking publish workflow status on $RELEASE_BRANCH..."
    LATEST_CONCLUSION="$(gh run list \
        --repo "$REPO_SLUG" \
        --workflow=publish.yml \
        --branch="$RELEASE_BRANCH" \
        --limit=1 \
        --json conclusion \
        -q '.[0].conclusion // "none"' 2>/dev/null || echo "none")"

    if [[ "$LATEST_CONCLUSION" == "success" ]]; then
        log "Publish workflow passed on $RELEASE_BRANCH"
    elif [[ "$LATEST_CONCLUSION" == "none" ]]; then
        echo "WARNING: No publish workflow runs found for $RELEASE_BRANCH" >&2
        read -p "Continue without publish verification? [y/N] " confirm
        [[ "$confirm" =~ ^[Yy]$ ]] || die "aborted"
    else
        echo "WARNING: Latest publish workflow on $RELEASE_BRANCH concluded: $LATEST_CONCLUSION" >&2
        read -p "Continue despite workflow status? [y/N] " confirm
        [[ "$confirm" =~ ^[Yy]$ ]] || die "aborted"
    fi
fi

# --- tag the release ----------------------------------------------------------
log "Checking out $RELEASE_BRANCH..."
run git checkout "$RELEASE_BRANCH"

log "Creating signed tag $TAG..."
if ! $DRY_RUN; then
    git tag -s -m "Release $TAG" "$TAG"
fi

# --- push tag -----------------------------------------------------------------
if $NO_PUSH || $DRY_RUN; then
    log "Skipping push"
else
    log "Pushing tag $TAG to $REMOTE..."
    git push "$REMOTE" "$TAG"
fi

# --- merge the release PR -----------------------------------------------------
if $NO_PR || $DRY_RUN; then
    log "Skipping PR merge and GitHub Release"
else
    REPO_SLUG="$(gh repo view --json nameWithOwner -q .nameWithOwner)"

    log "Finding release PR..."
    PR_NUMBER="$(gh pr list \
        --repo "$REPO_SLUG" \
        --head "$RELEASE_BRANCH" \
        --base main \
        --state open \
        --json number \
        -q '.[0].number')"

    if [[ -z "$PR_NUMBER" ]]; then
        die "no open PR found from $RELEASE_BRANCH to main"
    fi

    log "Merging PR #${PR_NUMBER} with merge commit..."
    gh pr merge "$PR_NUMBER" \
        --repo "$REPO_SLUG" \
        --merge \
        --subject "Merge $RELEASE_BRANCH for release $TAG"

    log "Creating GitHub Release $TAG..."
    gh release create "$TAG" \
        --repo "$REPO_SLUG" \
        --title "Release $TAG" \
        --generate-notes
fi

# --- switch back to main ------------------------------------------------------
log "Switching back to main..."
run git checkout main

echo ""
log "Done. Release $TAG complete."
if ! $NO_PUSH && ! $DRY_RUN; then
    log "The GitHub Release will trigger the publish workflow."
fi
