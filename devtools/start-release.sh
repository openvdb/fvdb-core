#!/usr/bin/env bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# Start a release: create release branch, bump versions, open merge PR.
# For hotfix releases (PATCH > 0): prepare the release branch for cherry-picks.
#
# Usage: ./devtools/start-release.sh <version> [options]
# Example: ./devtools/start-release.sh 0.4.0 --remote upstream
#          ./devtools/start-release.sh 0.4.1  (hotfix release)

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

Start a release burndown by creating a release branch, setting versions, and
opening a merge PR. When PATCH > 0, prepares the existing release branch for
a hotfix release instead.

Arguments:
  <version>       Release version in MAJOR.MINOR.PATCH format (e.g. 0.4.0)
                  Use PATCH > 0 for hotfix releases (e.g. 0.4.1)

Options:
  --remote NAME   Git remote to push to (default: upstream)
  --dry-run       Print what would happen without making changes
  --no-push       Skip git push (for testing)
  --no-pr         Skip PR creation (for testing)
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

# Parse a MAJOR.MINOR.PATCH version string and compute the next dev version.
# Bumps MINOR by 1, resets PATCH to 0, appends .dev0.
next_dev_version() {
    local ver="$1"
    local major minor patch
    IFS='.' read -r major minor patch <<< "$ver"
    echo "${major}.$((minor + 1)).0.dev0"
}

# Extract the MAJOR.MINOR prefix from a version string.
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

# Update the version in pyproject.toml.
set_version() {
    local new_version="$1"
    sed -i "s/^version = \".*\"/version = \"${new_version}\"/" "$REPO_ROOT/pyproject.toml"
}

# Read the current version from pyproject.toml.
get_version() {
    grep '^version = ' "$REPO_ROOT/pyproject.toml" | sed 's/^version = "\(.*\)"/\1/'
}

# Extract the PATCH number from a MAJOR.MINOR.PATCH version string.
extract_patch() {
    local ver="$1"
    local _major _minor patch
    IFS='.' read -r _major _minor patch <<< "$ver"
    echo "$patch"
}

# Return the tag name for the release immediately before this version.
previous_patch_tag() {
    local ver="$1"
    local major minor patch
    IFS='.' read -r major minor patch <<< "$ver"
    echo "v${major}.${minor}.$((patch - 1))"
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

# Validate version format
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "version must be MAJOR.MINOR.PATCH (got: $VERSION)"

# --- derived values -----------------------------------------------------------
BRANCH_SUFFIX="$(release_branch_suffix "$VERSION")"
RELEASE_BRANCH="release/v${BRANCH_SUFFIX}"
NEXT_DEV="$(next_dev_version "$VERSION")"
PATCH="$(extract_patch "$VERSION")"
IS_HOTFIX=false
[[ "$PATCH" -gt 0 ]] && IS_HOTFIX=true

# --- hotfix release -----------------------------------------------------------
if $IS_HOTFIX; then
    PREV_TAG="$(previous_patch_tag "$VERSION")"

    cd "$REPO_ROOT"

    log "Hotfix version:      $VERSION"
    log "Previous tag:        $PREV_TAG"
    log "Release branch:      $RELEASE_BRANCH"
    log "Remote:              $REMOTE"
    echo ""

    if ! $DRY_RUN; then
        if ! git diff --quiet || ! git diff --cached --quiet; then
            die "working tree is not clean; commit or stash changes first"
        fi

        if ! $NO_PUSH; then
            log "Fetching $REMOTE..."
            git fetch "$REMOTE"
        fi

        git rev-parse "${PREV_TAG}^{commit}" >/dev/null 2>&1 \
            || die "previous release tag $PREV_TAG not found"

        git show-ref --verify --quiet "refs/heads/$RELEASE_BRANCH" \
            || die "branch $RELEASE_BRANCH does not exist locally"

        BRANCH_COMMIT="$(git rev-parse "$RELEASE_BRANCH")"
        TAG_COMMIT="$(git rev-parse "${PREV_TAG}^{commit}")"
        if [[ "$BRANCH_COMMIT" != "$TAG_COMMIT" ]]; then
            die "$RELEASE_BRANCH ($(git rev-parse --short "$RELEASE_BRANCH")) is not at $PREV_TAG ($(git rev-parse --short "${PREV_TAG}^{commit}")); the branch must match the previous release tag"
        fi
    fi

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    # The release branch may predate devtools/ scripts, so save a copy of
    # companion scripts before the checkout replaces the working tree.
    _HOTFIX_TMPDIR="$(mktemp -d)"
    trap 'rm -rf "$_HOTFIX_TMPDIR"' EXIT
    cp "$SCRIPT_DIR/update-doc-versions.sh" "$_HOTFIX_TMPDIR/" 2>/dev/null || true
    chmod +x "$_HOTFIX_TMPDIR/update-doc-versions.sh" 2>/dev/null || true

    log "Checking out $RELEASE_BRANCH..."
    run git checkout "$RELEASE_BRANCH"

    if ! $DRY_RUN; then
        CURRENT_VER="$(get_version)"
        if [[ "$CURRENT_VER" != "$VERSION" ]]; then
            log "Setting version to $VERSION on $RELEASE_BRANCH..."
            set_version "$VERSION"
        else
            log "Version already set to $VERSION on $RELEASE_BRANCH"
        fi

        if [[ -x "$_HOTFIX_TMPDIR/update-doc-versions.sh" ]]; then
            "$_HOTFIX_TMPDIR/update-doc-versions.sh" "$VERSION"
        fi

        if ! git diff --quiet || ! git diff --cached --quiet; then
            git add pyproject.toml
            [[ -f docs/conf.py ]] && git add docs/conf.py
            git commit -s -S -m "Set version to $VERSION for hotfix release"
        fi
    fi

    rm -rf "$_HOTFIX_TMPDIR"

    echo ""
    log "Done. Hotfix v${VERSION} prepared on $RELEASE_BRANCH."
    echo ""
    log "Next steps:"
    log "  1. Cherry-pick or apply fixes to $RELEASE_BRANCH"
    log "  2. Push $RELEASE_BRANCH to $REMOTE"
    log "  3. Wait for publish workflow to pass"
    log "  4. Run: ./devtools/finish-release.sh $VERSION"
    exit 0
fi

# --- pre-flight checks -------------------------------------------------------
cd "$REPO_ROOT"

log "Release version:     $VERSION"
log "Release branch:      $RELEASE_BRANCH"
log "Next dev version:    $NEXT_DEV"
log "Remote:              $REMOTE"
echo ""

if ! $DRY_RUN; then
    CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
    [[ "$CURRENT_BRANCH" == "main" || "$CURRENT_BRANCH" == "$RELEASE_BRANCH" ]] \
        || die "must be on main or $RELEASE_BRANCH (currently on $CURRENT_BRANCH)"

    if ! git diff --quiet || ! git diff --cached --quiet; then
        die "working tree is not clean; commit or stash changes first"
    fi

    if ! $NO_PUSH; then
        log "Fetching $REMOTE..."
        git fetch "$REMOTE"
        assert_branch_current main
    fi

    if [[ "$CURRENT_BRANCH" != "main" ]]; then
        git checkout main
    fi
fi

# --- create release branch and set version ------------------------------------
if ! $DRY_RUN && git show-ref --verify --quiet "refs/heads/$RELEASE_BRANCH"; then
    log "Branch $RELEASE_BRANCH already exists locally, checking out..."
    run git checkout "$RELEASE_BRANCH"
else
    log "Creating branch $RELEASE_BRANCH from main..."
    run git checkout -b "$RELEASE_BRANCH"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! $DRY_RUN; then
    CURRENT_VER="$(get_version)"
    if [[ "$CURRENT_VER" != "$VERSION" ]]; then
        log "Setting version to $VERSION on $RELEASE_BRANCH..."
        set_version "$VERSION"
    else
        log "Version already set to $VERSION on $RELEASE_BRANCH"
    fi

    "$SCRIPT_DIR/update-doc-versions.sh" "$VERSION"

    if ! git diff --quiet || ! git diff --cached --quiet; then
        git add pyproject.toml
        [[ -f docs/conf.py ]] && git add docs/conf.py
        git commit -s -S -m "Set version to $VERSION for release"
    fi
fi

# --- switch back to main and bump to next dev version -------------------------
log "Switching back to main..."
run git checkout main

if ! $DRY_RUN; then
    CURRENT_VER="$(get_version)"
    if [[ "$CURRENT_VER" == "$NEXT_DEV" ]]; then
        log "Main already at $NEXT_DEV"
    else
        log "Bumping main version to $NEXT_DEV..."
        set_version "$NEXT_DEV"
        git add pyproject.toml
        git commit -s -S -m "Bump version to $NEXT_DEV after $RELEASE_BRANCH branch"
    fi
fi

# --- push ---------------------------------------------------------------------
if $NO_PUSH || $DRY_RUN; then
    log "Skipping push"
else
    log "Pushing main to $REMOTE..."
    git push "$REMOTE" main

    log "Pushing $RELEASE_BRANCH to $REMOTE..."
    git push "$REMOTE" "$RELEASE_BRANCH"
fi

# --- open PR ------------------------------------------------------------------
if $NO_PR || $DRY_RUN; then
    log "Skipping PR creation"
else
    REPO_SLUG="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
    EXISTING_PR="$(gh pr list --repo "$REPO_SLUG" --head "$RELEASE_BRANCH" --base main --state open --json number -q '.[0].number' 2>/dev/null || true)"

    if [[ -n "$EXISTING_PR" ]]; then
        log "Release PR #${EXISTING_PR} already exists"
    else
        log "Creating draft PR from $RELEASE_BRANCH to main..."

        PR_BODY="$(cat <<EOF
## Release v${VERSION}

Track the release burndown for \`${RELEASE_BRANCH}\`.

**Note:** This PR is intentionally a draft and will **not** be merged directly.
\`finish-release.sh\` will close this PR and create an \`adopt/v${BRANCH_SUFFIX}\`
branch that reconciles the version in \`pyproject.toml\` before merging into
\`main\`.

### Checklist

- [ ] Publish workflow passing on \`${RELEASE_BRANCH}\` (builds + smoke tests)
- [ ] All planned fixes merged into \`${RELEASE_BRANCH}\`
- [ ] Code freeze applied (branch protection tightened)
- [ ] Release notes drafted

### Release command

\`\`\`bash
./devtools/finish-release.sh ${VERSION}
\`\`\`
EOF
)"
        gh pr create \
            --repo "$REPO_SLUG" \
            --base main \
            --head "$RELEASE_BRANCH" \
            --title "Release v${VERSION}" \
            --body "$PR_BODY" \
            --draft
    fi
fi

echo ""
log "Done. Release burndown started for v${VERSION}."
if ! $NO_PUSH && ! $DRY_RUN; then
    REPO_SLUG="$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || echo "unknown")"
    log "The push to $RELEASE_BRANCH will trigger the publish workflow."
    log "Staging wheels will be built and validated automatically."
    log "Monitor: https://github.com/${REPO_SLUG}/actions/workflows/publish.yml"
fi
echo ""
log "Next steps:"
log "  1. Select open PRs to retarget to $RELEASE_BRANCH, or submit bug fix PRs"
log "  2. Apply code freeze when ready"
log "  3. Run: ./devtools/finish-release.sh $VERSION"
