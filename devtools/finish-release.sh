#!/usr/bin/env bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# Finish a release: tag, create adopt branch, open adopt PR, create GitHub Release.
# Works for both minor releases (PATCH == 0) and hotfix releases (PATCH > 0).
#
# Usage: ./devtools/finish-release.sh <version> [options]
# Example: ./devtools/finish-release.sh 0.4.0 --remote upstream
#          ./devtools/finish-release.sh 0.4.1  (hotfix release)

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

Finish a release by tagging the release branch, creating an adopt branch to
reconcile versions, and creating a GitHub Release. Works for both minor
releases (PATCH == 0) and hotfix releases (PATCH > 0).

Arguments:
  <version>       Release version in MAJOR.MINOR.PATCH format (e.g. 0.4.0)
                  Use PATCH > 0 for hotfix releases (e.g. 0.4.1)

Options:
  --remote NAME   Git remote to push to (default: upstream)
  --dry-run       Print what would happen without making changes
  --no-push       Skip git push and GitHub operations (for testing)
  --no-pr         Skip PR close/create and GitHub Release (for testing)
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

assert_branch_current() {
    local branch="$1"
    local remote_ref="$REMOTE/$branch"
    if ! git show-ref --verify --quiet "refs/remotes/$remote_ref"; then
        die "$branch not found on $REMOTE; push it first"
    fi
    local local_rev remote_rev
    local_rev="$(git rev-parse --short "$branch")"
    remote_rev="$(git rev-parse --short "$remote_ref")"
    if [[ "$local_rev" != "$remote_rev" ]]; then
        die "$branch ($local_rev) differs from $remote_ref ($remote_rev); pull or reset first"
    fi
}

get_version_from_ref() {
    local ref="$1"
    git show "${ref}:pyproject.toml" | grep '^version = ' | sed 's/^version = "\(.*\)"/\1/'
}

set_version() {
    local new_version="$1"
    sed -i "s/^version = \".*\"/version = \"${new_version}\"/" "$REPO_ROOT/pyproject.toml"
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

_PATCH="${VERSION##*.}"
IS_HOTFIX=false
[[ "$_PATCH" -gt 0 ]] && IS_HOTFIX=true

if $IS_HOTFIX; then
    ADOPT_BRANCH="adopt/v${VERSION}"
else
    ADOPT_BRANCH="adopt/v${BRANCH_SUFFIX}"
fi

# --- pre-flight checks -------------------------------------------------------
cd "$REPO_ROOT"

log "Release version:     $VERSION"
log "Release branch:      $RELEASE_BRANCH"
log "Adopt branch:        $ADOPT_BRANCH"
log "Tag:                 $TAG"
log "Remote:              $REMOTE"
echo ""

if ! $DRY_RUN; then
    if ! git diff --quiet || ! git diff --cached --quiet; then
        die "working tree is not clean; commit or stash changes first"
    fi

    if ! git show-ref --verify --quiet "refs/heads/$RELEASE_BRANCH"; then
        die "branch $RELEASE_BRANCH does not exist locally"
    fi

    if ! $NO_PUSH; then
        log "Fetching $REMOTE..."
        git fetch "$REMOTE"
        assert_branch_current "$RELEASE_BRANCH"
        assert_branch_current main
    fi

    if git rev-parse "$TAG" >/dev/null 2>&1; then
        die "tag $TAG already exists"
    fi

    if git show-ref --verify --quiet "refs/heads/$ADOPT_BRANCH"; then
        die "branch $ADOPT_BRANCH already exists; delete it first if re-running"
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

# --- create adopt branch with version fixup -----------------------------------
log "Creating $ADOPT_BRANCH from $RELEASE_BRANCH..."
run git checkout -b "$ADOPT_BRANCH" "$RELEASE_BRANCH"

if ! $DRY_RUN; then
    MAIN_VERSION="$(get_version_from_ref main)"
    log "Setting version to $MAIN_VERSION (from main) on $ADOPT_BRANCH..."
    set_version "$MAIN_VERSION"
    git add pyproject.toml
    git commit -s -S -m "Set version to $MAIN_VERSION for merge into main"
fi

# --- push adopt branch --------------------------------------------------------
if $NO_PUSH || $DRY_RUN; then
    log "Skipping adopt branch push"
else
    log "Pushing $ADOPT_BRANCH to $REMOTE..."
    git push "$REMOTE" "$ADOPT_BRANCH"
fi

# --- close release PR and open adopt PR ---------------------------------------
if $NO_PR || $DRY_RUN; then
    log "Skipping PR operations and GitHub Release"
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

    if [[ -n "$PR_NUMBER" ]]; then
        log "Closing release PR #${PR_NUMBER}..."
        gh pr close "$PR_NUMBER" \
            --repo "$REPO_SLUG" \
            --comment "Closed by finish-release.sh. Release changes will be merged via $ADOPT_BRANCH."
    else
        log "No open release PR found from $RELEASE_BRANCH to main (skipping close)"
    fi

    log "Creating PR from $ADOPT_BRANCH to main..."

    if $IS_HOTFIX; then
        ADOPT_PR_BODY="$(cat <<EOF
## Adopt hotfix v${VERSION}

Merge hotfix changes from \`${RELEASE_BRANCH}\` into \`main\` via the
\`${ADOPT_BRANCH}\` branch. The version in \`pyproject.toml\` has been set to
the current \`main\` development version to avoid conflicts.

**Tag:** \`${TAG}\`

### To merge

Merge this PR once CI passes. Use a **merge commit** (not squash) to preserve
the release branch history on \`main\`.
EOF
)"
    else
        ADOPT_PR_BODY="$(cat <<EOF
## Adopt release v${VERSION}

Merge release changes from \`${RELEASE_BRANCH}\` into \`main\` via the
\`${ADOPT_BRANCH}\` branch. The version in \`pyproject.toml\` has been set to
the current \`main\` development version to avoid conflicts.

**Tag:** \`${TAG}\`

This PR replaces the original release PR${PR_NUMBER:+ (#${PR_NUMBER})} which was
closed because \`${RELEASE_BRANCH}\` intentionally carries a different version
string (\`${VERSION}\`) than \`main\`.

### To merge

Merge this PR once CI passes. Use a **merge commit** (not squash) to preserve
the release branch history on \`main\`.
EOF
)"
    fi

    ADOPT_KIND="release"
    $IS_HOTFIX && ADOPT_KIND="hotfix"

    gh pr create \
        --repo "$REPO_SLUG" \
        --base main \
        --head "$ADOPT_BRANCH" \
        --title "Adopt $ADOPT_KIND v${VERSION}" \
        --body "$ADOPT_PR_BODY"

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
log "Done. Release $TAG tagged."
if ! $NO_PR && ! $DRY_RUN; then
    log "The GitHub Release will trigger the publish workflow."
    log "Merge the adopt PR from $ADOPT_BRANCH once CI passes."
    log "After merging, re-deploy docs: gh workflow run docs.yml --ref main"
fi
