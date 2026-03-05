# Release Process

fVDB uses a simplified [OneFlow](https://www.endoflineblog.com/oneflow-a-git-branching-model-and-workflow)
branching model with short-lived release branches.

## Branching Model

`main` is the single long-lived default branch. It always contains the latest
development code. Release branches (`release/vX.Y`) are created for stabilization
and are merged back into `main` at release time.

```
main:    ──A──B──C─────────────F──G──H── ...
               \               /
release/v0.4:   D──E──────────T (tag v0.4.0)
```

- **A, B, C**: normal development on `main`
- **C**: burndown begins; `release/v0.4` branches off, `main` version bumped
- **D, E**: bug fixes on the release branch during burndown
- **T**: release tag created on the release branch
- **F**: merge commit bringing release fixes back into `main`
- **G, H**: development continues

## Branch Naming

| Purpose | Pattern | Example |
|---------|---------|---------|
| Release | `release/vMAJOR.MINOR` | `release/v0.4` |
| Hotfix | `hotfix/vMAJOR.MINOR.PATCH` | `hotfix/v0.4.1` |

## Version Management

The version in `pyproject.toml` is the single source of truth.

| Branch | Version | Example |
|--------|---------|---------|
| `release/vX.Y` | `X.Y.0` | `0.4.0` |
| `main` (after branching) | `X.Z.0.dev0` | `0.5.0.dev0` |

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
| **Release day** | Day ~7 | Tag, merge PR, create GitHub Release |

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
5. Open a PR from `release/v0.4` into `main`

Use `--remote origin` to push to a different remote.
Use `--dry-run` to preview without making changes.

### During Burndown

Once the release branch exists, PR targeting rules change:

- **New PRs opened during burndown** target `main` by default and will ship in
  the *following* release. If a PR is needed in the current release, it must
  target the release branch directly and requires maintainer approval.
- **PRs opened before burndown** that have not yet merged can still land on
  `main`. If the change is also needed in the current release, cherry-pick the
  squash-merged commit from `main` onto the release branch:
  ```bash
  git checkout release/v0.4
  git cherry-pick <commit-sha>
  ```
  This may cause a minor conflict when the release branch is merged back into
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
1. Tag `v0.4.0` on the HEAD of `release/v0.4`
2. Push the tag
3. Merge the release PR into `main` (merge commit, not squash)
4. Create a GitHub Release (triggers the publish workflow)

Use `--remote origin` to target a different remote.
Use `--dry-run` to preview without making changes.

### Hotfixes

For critical fixes after a release:

1. Create a hotfix branch from the release tag:
   ```bash
   git checkout -b hotfix/v0.4.1 v0.4.0
   ```
2. Apply the fix and commit.
3. Open a PR from `hotfix/v0.4.1` to `main`.
4. If `release/v0.4` still exists, also open a PR to it.
5. Tag `v0.4.1` and create a GitHub Release.

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
| `devtools/start-release.sh` | Create release branch, bump versions, open PR |
| `devtools/finish-release.sh` | Tag release, merge PR, create GitHub Release |
| `devtools/test-release-scripts.sh` | End-to-end tests for the release scripts |

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
