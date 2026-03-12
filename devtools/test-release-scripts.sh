#!/usr/bin/env bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end tests for start-release.sh and finish-release.sh.
# Runs against a disposable temporary git repo -- no network access needed.
#
# Usage: ./devtools/test-release-scripts.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PASS_COUNT=0
FAIL_COUNT=0

pass() {
    PASS_COUNT=$((PASS_COUNT + 1))
    echo "  PASS: $1"
}

fail() {
    FAIL_COUNT=$((FAIL_COUNT + 1))
    echo "  FAIL: $1" >&2
}

assert_eq() {
    local desc="$1" expected="$2" actual="$3"
    if [[ "$expected" == "$actual" ]]; then
        pass "$desc"
    else
        fail "$desc (expected '$expected', got '$actual')"
    fi
}

assert_contains() {
    local desc="$1" haystack="$2" needle="$3"
    if [[ "$haystack" == *"$needle"* ]]; then
        pass "$desc"
    else
        fail "$desc (expected to contain '$needle')"
    fi
}

assert_branch_exists() {
    local desc="$1" branch="$2"
    if git show-ref --verify --quiet "refs/heads/$branch"; then
        pass "$desc"
    else
        fail "$desc (branch '$branch' not found)"
    fi
}

assert_tag_exists() {
    local desc="$1" tag="$2"
    if git rev-parse "$tag" >/dev/null 2>&1; then
        pass "$desc"
    else
        fail "$desc (tag '$tag' not found)"
    fi
}

get_version() {
    grep '^version = ' pyproject.toml | sed 's/^version = "\(.*\)"/\1/'
}

# --- setup: create a disposable test repo ------------------------------------
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

BARE_REPO="$TMPDIR/remote.git"
WORK_REPO="$TMPDIR/work"

git init --bare "$BARE_REPO" >/dev/null 2>&1

git clone "$BARE_REPO" "$WORK_REPO" >/dev/null 2>&1
cd "$WORK_REPO"

git config user.name "Test User"
git config user.email "test@example.com"
# Allow unsigned commits/tags in the test environment
git config commit.gpgsign false
git config tag.gpgsign false

# Copy the release scripts into the temp repo so REPO_ROOT resolves correctly
mkdir -p devtools
cp "$SCRIPT_DIR/start-release.sh" devtools/
cp "$SCRIPT_DIR/finish-release.sh" devtools/
chmod +x devtools/start-release.sh devtools/finish-release.sh
START_RELEASE="$WORK_REPO/devtools/start-release.sh"
FINISH_RELEASE="$WORK_REPO/devtools/finish-release.sh"

cat > pyproject.toml <<'PYPROJECT'
[project]
name = "fvdb-core"
version = "0.3.1"
PYPROJECT

git add pyproject.toml
git commit -s -m "Initial commit" >/dev/null 2>&1
git tag v0.3.0
git push origin main >/dev/null 2>&1
git push origin v0.3.0 >/dev/null 2>&1

echo ""
echo "============================================="
echo " Test: start-release.sh --help"
echo "============================================="

HELP_OUTPUT="$("$START_RELEASE" --help 2>&1)"
assert_contains "help text mentions version" "$HELP_OUTPUT" "MAJOR.MINOR.PATCH"
assert_contains "help text mentions --dry-run" "$HELP_OUTPUT" "--dry-run"

echo ""
echo "============================================="
echo " Test: start-release.sh --dry-run"
echo "============================================="

DRY_OUTPUT="$("$START_RELEASE" 0.4.0 --dry-run 2>&1)"
assert_contains "dry-run mentions release branch" "$DRY_OUTPUT" "release/v0.4"
assert_contains "dry-run mentions next dev version" "$DRY_OUTPUT" "0.5.0.dev0"

BRANCHES_BEFORE="$(git branch --list)"
assert_eq "no branches created in dry-run" "$(echo "$BRANCHES_BEFORE" | grep -c 'release/v0.4' || true)" "0"

VERSION_AFTER_DRY="$(get_version)"
assert_eq "version unchanged after dry-run" "0.3.1" "$VERSION_AFTER_DRY"

echo ""
echo "============================================="
echo " Test: start-release.sh (no-push, no-pr)"
echo "============================================="

START_OUTPUT="$("$START_RELEASE" 0.4.0 --no-push --no-pr 2>&1)"

assert_branch_exists "release/v0.4 branch exists" "release/v0.4"
assert_contains "next steps mention retarget" "$START_OUTPUT" "retarget"

CURRENT="$(git rev-parse --abbrev-ref HEAD)"
assert_eq "on main after start-release" "main" "$CURRENT"

MAIN_VERSION="$(get_version)"
assert_eq "main version is 0.5.0.dev0" "0.5.0.dev0" "$MAIN_VERSION"

git checkout release/v0.4 >/dev/null 2>&1
RELEASE_VERSION="$(get_version)"
assert_eq "release branch version is 0.4.0" "0.4.0" "$RELEASE_VERSION"

RELEASE_LOG="$(git log -1 --format='%B' release/v0.4)"
assert_contains "release commit has DCO sign-off" "$RELEASE_LOG" "Signed-off-by:"

MAIN_LOG="$(git log -1 --format='%B' main)"
assert_contains "main bump commit has DCO sign-off" "$MAIN_LOG" "Signed-off-by:"

git checkout main >/dev/null 2>&1

echo ""
echo "============================================="
echo " Test: start-release.sh is idempotent"
echo "============================================="

RERUN_OUTPUT="$("$START_RELEASE" 0.4.0 --no-push --no-pr 2>&1)"
pass "re-running start-release succeeds"
assert_contains "re-run detects existing branch" "$RERUN_OUTPUT" "already exists"
assert_contains "re-run detects version already set" "$RERUN_OUTPUT" "already"

MAIN_VERSION_RERUN="$(get_version)"
assert_eq "main still at 0.5.0.dev0 after re-run" "0.5.0.dev0" "$MAIN_VERSION_RERUN"

git checkout release/v0.4 >/dev/null 2>&1
RELEASE_VERSION_RERUN="$(get_version)"
assert_eq "release branch still at 0.4.0 after re-run" "0.4.0" "$RELEASE_VERSION_RERUN"
git checkout main >/dev/null 2>&1

echo ""
echo "============================================="
echo " Test: start-release.sh validates version"
echo "============================================="

if "$START_RELEASE" "bad.version" --no-push --no-pr 2>&1; then
    fail "should reject invalid version format"
else
    pass "rejects invalid version format"
fi

echo ""
echo "============================================="
echo " Test: finish-release.sh --help"
echo "============================================="

FHELP_OUTPUT="$("$FINISH_RELEASE" --help 2>&1)"
assert_contains "finish help mentions version" "$FHELP_OUTPUT" "MAJOR.MINOR.PATCH"
assert_contains "finish help mentions --dry-run" "$FHELP_OUTPUT" "--dry-run"

echo ""
echo "============================================="
echo " Test: finish-release.sh --dry-run"
echo "============================================="

FDRY_OUTPUT="$("$FINISH_RELEASE" 0.4.0 --dry-run 2>&1)"
assert_contains "finish dry-run mentions tag" "$FDRY_OUTPUT" "v0.4.0"
assert_contains "finish dry-run mentions release branch" "$FDRY_OUTPUT" "release/v0.4"
assert_contains "finish dry-run mentions adopt branch" "$FDRY_OUTPUT" "adopt/v0.4"

echo ""
echo "============================================="
echo " Test: finish-release.sh (no-push, no-pr)"
echo "============================================="

"$FINISH_RELEASE" 0.4.0 --no-push --no-pr 2>&1

assert_tag_exists "tag v0.4.0 exists" "v0.4.0"

TAG_COMMIT="$(git rev-parse 'v0.4.0^{commit}')"
RELEASE_HEAD="$(git rev-parse release/v0.4)"
assert_eq "tag points to release branch HEAD" "$RELEASE_HEAD" "$TAG_COMMIT"

CURRENT_AFTER="$(git rev-parse --abbrev-ref HEAD)"
assert_eq "on main after finish-release" "main" "$CURRENT_AFTER"

assert_branch_exists "adopt/v0.4 branch exists" "adopt/v0.4"

git checkout adopt/v0.4 >/dev/null 2>&1
ADOPT_VERSION="$(get_version)"
assert_eq "adopt branch version is 0.5.0.dev0" "0.5.0.dev0" "$ADOPT_VERSION"

ADOPT_LOG="$(git log -1 --format='%B' adopt/v0.4)"
assert_contains "adopt commit has DCO sign-off" "$ADOPT_LOG" "Signed-off-by:"
git checkout main >/dev/null 2>&1

git checkout release/v0.4 >/dev/null 2>&1
RELEASE_VERSION_AFTER="$(get_version)"
assert_eq "release branch still at 0.4.0 after finish" "0.4.0" "$RELEASE_VERSION_AFTER"
git checkout main >/dev/null 2>&1

echo ""
echo "============================================="
echo " Test: adopt branch merges cleanly into main"
echo "============================================="

MERGE_OUTPUT="$(git merge --no-commit --no-ff adopt/v0.4 2>&1)"
MERGE_EXIT=$?
git reset --hard HEAD >/dev/null 2>&1

if [[ $MERGE_EXIT -eq 0 ]]; then
    pass "adopt branch merges cleanly into main"
else
    fail "adopt branch has merge conflicts with main: $MERGE_OUTPUT"
fi

echo ""
echo "============================================="
echo " Test: finish-release.sh rejects duplicate tag"
echo "============================================="

if "$FINISH_RELEASE" 0.4.0 --no-push --no-pr 2>&1; then
    fail "should reject when tag already exists"
else
    pass "rejects duplicate tag"
fi

# --- summary ------------------------------------------------------------------
echo ""
echo "============================================="
echo " Results: $PASS_COUNT passed, $FAIL_COUNT failed"
echo "============================================="

if [[ $FAIL_COUNT -gt 0 ]]; then
    exit 1
fi
