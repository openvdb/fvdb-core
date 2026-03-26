# Release Process

fVDB uses a simplified [OneFlow](https://www.endoflineblog.com/oneflow-a-git-branching-model-and-workflow)
branching model with short-lived release branches.

## Branching Model

`main` is the single long-lived default branch. It always contains the latest
development code. Release branches (`release/vX.Y`) are created for
stabilization. At release time, an adopt branch (`adopt/vX.Y`) reconciles
the version in `pyproject.toml` and merges the release work back into `main`,
keeping the release branch available for future hotfix releases. Hotfix
releases (PATCH > 0) reuse the same release branch with a version-specific
adopt branch (`adopt/vX.Y.Z`).

```
main:           ──A──B──C──D──────────G──H──────────────K── ...
                       \             /       \          /
release/v0.4:           E──F──T─────/─────P──Q──U      /
                               \   /              \   /
adopt/v0.4:                     V─┘                \ /
adopt/v0.4.1:                                       W
                        (v0.4.0)             (v0.4.1)
```

- **A, B**: normal development on `main`
- **B**: burndown begins; `release/v0.4` branches off
- **C**: `main` version bumped to next `.dev0`
- **D**: new feature merged to `main` (not included in the release)
- **E, F**: bug fixes or pre-burndown PRs merged to the release branch
- **T**: release tag `v0.4.0` created on the release branch
- **V**: adopt branch created from `T`; version set to match `main`
- **G**: merge commit bringing release fixes back into `main` via `adopt/v0.4`
- **H**: development continues
- **P**: version bumped to `0.4.1` on the release branch (`start-release.sh 0.4.1`)
- **Q**: cherry-picked fix
- **U**: release tag `v0.4.1` created on the release branch
- **W**: adopt branch `adopt/v0.4.1` created from `U`; version set to match `main`
- **K**: merge commit bringing hotfix into `main` via `adopt/v0.4.1`

## Branch Naming

| Purpose | Pattern | Example |
|---------|---------|---------|
| Release | `release/vMAJOR.MINOR` | `release/v0.4` |
| Adopt | `adopt/vMAJOR.MINOR` | `adopt/v0.4` |
| Adopt (hotfix) | `adopt/vMAJOR.MINOR.PATCH` | `adopt/v0.4.1` |

## Version Management

The version in `pyproject.toml` is the single source of truth.

| Branch | Version | Example |
|--------|---------|---------|
| `release/vX.Y` | `X.Y.0` | `0.4.0` |
| `release/vX.Y` (hotfix) | `X.Y.Z` | `0.4.1` |
| `main` (after branching) | `X.Z.0.dev0` | `0.5.0.dev0` |
| `adopt/vX.Y` | `X.Z.0.dev0` (matches `main`) | `0.5.0.dev0` |
| `adopt/vX.Y.Z` | `X.Z.0.dev0` (matches `main`) | `0.5.0.dev0` |

The `.dev0` suffix on `main` is a [PEP 440](https://peps.python.org/pep-0440/)
pre-release marker that signals unreleased development code. It does not affect
nightly builds, which override the version entirely.

## Timeline

A typical release takes about one week:

| Phase | Duration | What happens |
|-------|----------|--------------|
| **Burndown start** | Day 0 | Create release branch, bump `main` version, open merge PR |
| **Burndown** | ~4-6 days | Stabilize: fix bugs on the release branch, continue features on `main` |
| **Code freeze** | 1-3 days | Only critical fixes on the release branch (admin merge only) |
| **Release day** | Day ~7 | Tag, create adopt branch + PR, create GitHub Release, merge adopt PR |

## Procedures

### Starting a Release (Burndown)

Run from a clean, up-to-date checkout of `main`:

```bash
./devtools/start-release.sh 0.4.0
```

This will:
1. Create `release/v0.4` from current `main`
2. Set the version to `0.4.0` on the release branch
3. Bump `main` to `0.5.0.dev0`
4. Push both branches to `upstream`
5. Open a **draft** PR from `release/v0.4` into `main` (tracks burndown;
   will be closed by `finish-release.sh` and replaced by an adopt PR)

Use `--remote origin` to push to a different remote.
Use `--dry-run` to preview without making changes.

### During Burndown

Once the release branch exists, PR targeting rules change:

- **New PRs opened during burndown** target `main` by default and will ship in
  the *following* release. If a PR is needed in the current release, it must
  target the release branch directly and requires maintainer approval.
- **PRs opened before burndown** that have not yet merged should be triaged
  during burndown planning. Decide whether each open PR is needed in the current
  release and retarget it to the release branch if so. After merging, if the
  change is also needed immediately on `main`, cherry-pick the squash-merged commit
  from the release branch back onto `main`:
  ```bash
  git checkout main
  git cherry-pick <commit-sha>
  ```
  This may cause a minor conflict when the adopt branch is merged back into
  `main`, but the content will be nearly identical and easy to resolve.
- **New features** continue targeting `main` as usual.

### Code Freeze

Tighten branch protection on `release/v0.4`:
- No new PRs to the release branch except approved critical fixes
- Require admin approval for merges

This is a manual GitHub settings change, not automated by the scripts.

### Finishing a Release

On release day:

```bash
./devtools/finish-release.sh 0.4.0
```

This will:
1. Verify the publish workflow passed on the release branch
2. Tag `v0.4.0` on the HEAD of `release/v0.4`
3. Push the tag
4. Create `adopt/v0.4` from `release/v0.4` with a commit that sets the
   version in `pyproject.toml` to match `main` (e.g. `0.5.0.dev0`)
5. Push `adopt/v0.4`
6. Close the draft release PR
7. Open a new PR from `adopt/v0.4` into `main`
8. Create a GitHub Release (triggers the publish workflow)

After the script finishes, **merge the adopt PR** once CI passes. Use a
merge commit (not squash) to preserve the release branch history on `main`.
The `release/v0.4` branch is left untouched at version `0.4.0`, available
as the base for future hotfix releases.

After merging, **re-deploy the documentation** to GitHub Pages so it reflects
the new release version:

```bash
gh workflow run docs.yml --ref main
```

Or use the GitHub Actions UI: **Actions > fvdb-core Documentation > Run
workflow** (select `main`).

Use `--remote origin` to target a different remote.
Use `--dry-run` to preview without making changes.

### Hotfix Releases

The same scripts handle hotfix releases. Pass a version with PATCH > 0
(e.g. `0.4.1`) to enter hotfix mode.

1. **Start the hotfix:**
   ```bash
   ./devtools/start-release.sh 0.4.1
   ```
   This verifies that `release/v0.4` is at the `v0.4.0` tag, bumps the
   version to `0.4.1`, and leaves you on the release branch ready to apply
   fixes. For successive hotfixes (e.g. `0.4.2`), the script verifies the
   branch is at the previous hotfix tag (`v0.4.1`).

2. **Cherry-pick or apply fixes:**
   ```bash
   git cherry-pick <commit-sha>
   ```

3. **Push the release branch** to trigger the publish workflow:
   ```bash
   git push upstream release/v0.4
   ```

4. **Wait for CI**, then finish the hotfix:
   ```bash
   ./devtools/finish-release.sh 0.4.1
   ```
   This tags `v0.4.1`, creates `adopt/v0.4.1` with a version fixup, opens
   a PR into `main`, and creates a GitHub Release.

5. **Merge the adopt PR** once CI passes (use a merge commit, not squash).

6. **Re-deploy documentation** after the adopt PR is merged:
   ```bash
   gh workflow run docs.yml --ref main
   ```

## GitHub Branch Protection

Recommended settings for `release/v*` branches during burndown:

- Require pull request reviews before merging
- Require status checks to pass
- Require signed commits

During code freeze, additionally:

- Restrict who can push (admin only)
- Require approval from a code owner

## Script Reference

| Script | Purpose |
|--------|---------|
| `devtools/start-release.sh` | Create release branch (or prepare hotfix), bump versions, open PR |
| `devtools/finish-release.sh` | Tag release, create adopt branch + PR, create GitHub Release |
| `devtools/test-release-scripts.sh` | End-to-end tests for the release scripts (including hotfix) |

All scripts support `--help`, `--dry-run`, and `--remote <name>` flags.

### Using with Other Repositories

The scripts live in fvdb-core but work with any repository that has a
`version = "..."` line in `pyproject.toml`. They detect the target repository
from the current working directory (`git rev-parse --show-toplevel`).

To release fvdb-reality-capture, run the scripts from inside that repo:

```bash
cd /path/to/fvdb-reality-capture
/path/to/fvdb-core/devtools/start-release.sh 0.4.0
```

## Automated Release Validation

Both fvdb-core and fvdb-reality-capture have `publish.yml` workflows that
auto-trigger on pushes to `release/v*` branches. This provides continuous
validation during the burndown period.

### What Happens Automatically

When code is pushed to a release branch (e.g. `release/v0.4`):

1. **fvdb-core**: `publish.yml` builds wheels for all Python/Torch/CUDA
   combinations in the matrix, uploads them to the S3 staging index
   (`simple-staging`), and runs GPU validation (smoke test + unit tests) on
   each variant.
2. **fvdb-reality-capture**: `publish.yml` builds a pure-Python wheel,
   uploads it to S3 staging, then runs GPU validation that installs
   fvdb-core from S3 staging to test the two packages together.

### S3 Staging Index

Release candidate wheels are published to the staging index at:

```
https://fvdb-packages.s3.us-east-2.amazonaws.com/simple-staging/
```

This allows manual testing before the final release. Staging wheels older
than 30 days are automatically pruned.

### Cross-Repository Lockstep Releases

fvdb-core and fvdb-reality-capture are released in lockstep. The procedure
is:

1. **Start burndown in fvdb-core first**:
   ```bash
   cd /path/to/fvdb-core
   ./devtools/start-release.sh 0.4.0
   ```
   Wait for the push to trigger the publish workflow and confirm wheels
   appear in S3 staging.

2. **Start burndown in fvdb-reality-capture**:
   ```bash
   cd /path/to/fvdb-reality-capture
   /path/to/fvdb-core/devtools/start-release.sh 0.4.0
   ```
   The fvdb-reality-capture validation will install fvdb-core from S3
   staging, verifying cross-package compatibility.

3. **Finish fvdb-core first**, then fvdb-reality-capture:
   ```bash
   cd /path/to/fvdb-core
   ./devtools/finish-release.sh 0.4.0

   cd /path/to/fvdb-reality-capture
   /path/to/fvdb-core/devtools/finish-release.sh 0.4.0
   ```

   After each `finish-release.sh` run, merge the resulting `adopt/v*` PR
   on GitHub, deleting the `adopt/v*` branch once CI passes. We keep the
   corresponding `release/vX.Y` branch for any future hotfix releases
   (PATCH > 0), reusing the same release branch rather than creating
   separate hotfix branches.

4. **Re-deploy documentation** after both adopt PRs are merged:
   ```bash
   gh workflow run docs.yml --ref main
   ```

`finish-release.sh` checks that the latest `publish.yml` run on the release
branch succeeded before proceeding. If the workflow failed or hasn't run, it
prompts for confirmation.

### Workflow Dispatch

Both workflows also support manual triggering via `workflow_dispatch` with
options to select the publish target (`s3`, `testpypi`, `none`) and whether
to run GPU validation.
