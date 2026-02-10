<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright Contributors to the OpenVDB Project -->

# fvdb Worktree Tools

Developer tools for managing Git worktrees across fvdb-core and fvdb-reality-capture repositories. These tools enable parallel development workflows where you can have multiple branches checked out simultaneously.

## Why Worktrees?

Git worktrees allow you to have multiple branches checked out at the same time in different directories. This is useful for:

- Working on multiple features in parallel
- Having multiple AI agents working on different issues simultaneously
- Quick context switching without stashing changes
- Keeping your main branch clean while working on features

## Installation

### Quick Install

```bash
cd devtools/worktree
./install.sh
```

The install script will:
1. Copy the tools to `~/bin`
2. Add `~/bin` to your PATH (if needed)
3. Create a configuration file (`~/.fvdb-devtools.conf`)

### Manual Install

```bash
# Copy scripts
mkdir -p ~/bin
cp fvdb-open fvdb-issue fvdb-close ~/bin/
chmod +x ~/bin/fvdb-{open,issue,close}

# Add to PATH (add to ~/.bashrc or ~/.zshrc)
export PATH="$HOME/bin:$PATH"

# Create config file
cat > ~/.fvdb-devtools.conf << 'EOF'
FVDB_CORE_PATH=/path/to/fvdb-core
FVDB_RC_PATH=/path/to/fvdb-reality-capture
EOF
```

## Configuration

Create `~/.fvdb-devtools.conf` with your repository paths:

```bash
# Required: paths to your main repository clones
FVDB_CORE_PATH=/home/yourname/github/fvdb-core
FVDB_RC_PATH=/home/yourname/github/fvdb-reality-capture

# Optional: GitHub repository names (these are the defaults)
FVDB_CORE_GH_REPO=openvdb/fvdb
FVDB_RC_GH_REPO=openvdb/fvdb-reality-capture
```

Alternatively, set these as environment variables.

## Requirements

- **Git** - with worktree support (Git 2.5+)
- **Bash 4.0+** - scripts use Bash 4 features; macOS ships Bash 3.2, so install a newer version via `brew install bash`
- **GitHub CLI** (`gh`) - for `fvdb-issue` to fetch issue details
- **jq** - for `fvdb-issue` to parse JSON
- **Cursor IDE** - or set `FVDB_EDITOR_CMD` to your preferred editor (e.g., `code`)
- **Optional: `claude` CLI** - required only if you use the `--claude` flag with `fvdb-issue`
- **Linux recommended** - `fvdb-close` "in use" detection uses `/proc` on Linux; on macOS it falls back to `lsof` (slower, may miss some processes)

Install requirements:
```bash
# Ubuntu/Debian
sudo apt-get install jq
# GitHub CLI: https://cli.github.com/

# macOS
brew install bash jq gh
```

## Tools

### fvdb-open

Interactive launcher for opening worktrees in Cursor.

```bash
# Interactive mode - select worktrees from a menu
fvdb-open

# Direct mode - specify worktrees by branch name
fvdb-open --core=main --rc=feature-branch

# Help
fvdb-open --help
```

**Features:**
- Lists all existing worktrees for both repositories
- Create new worktrees on new or existing branches
- Opens both repositories in a single Cursor multi-root workspace

### fvdb-issue

Create a worktree for a specific GitHub issue with AI agent context.

```bash
# By issue number (auto-detects repository)
fvdb-issue 187

# By full URL
fvdb-issue https://github.com/openvdb/fvdb-reality-capture/issues/187

# Open in Claude Code instead of Cursor
fvdb-issue 187 --claude

# Specify fvdb-core branch to use
fvdb-issue 187 --core=my-feature

# Help
fvdb-issue --help
```

**Features:**
- Fetches issue title, description, labels, and assignees
- Creates a branch named `issue-{number}-{title-slug}`
- Creates `.cursor/rules/current-issue.md` with issue context for AI agents
- Warns if issue is closed or is actually a PR
- Copies a starter prompt to clipboard for Cursor
- Supports Claude Code with auto-start (`--claude`)

### fvdb-close

Remove worktrees safely.

```bash
# Interactive mode - shows all worktrees to choose from
fvdb-close

# By name (partial match supported)
fvdb-close issue-187
fvdb-close mcmc

# By full path
fvdb-close /path/to/worktree

# Help
fvdb-close --help
```

**Features:**
- Lists all worktrees from both repositories
- Detects if worktree is in use (terminals with cwd there)
- Prompts to delete the branch after removing worktree
- Handles detached HEAD worktrees gracefully

## Workflow Examples

### Working on a GitHub Issue

```bash
# 1. Start working on issue #187
fvdb-issue 187

# 2. Cursor opens with issue context in .cursor/rules/current-issue.md
# 3. Paste the clipboard prompt to start the AI agent
# 4. Work on the issue...

# 5. When done, close the worktree
fvdb-close issue-187
# Optionally delete the branch when prompted
```

### Multiple Parallel Agents

```bash
# Terminal 1: Work on issue #100
fvdb-issue 100 --claude

# Terminal 2: Work on issue #200
fvdb-issue 200 --claude

# Terminal 3: Manual work on a feature
fvdb-open --core=main --rc=my-feature
```

### Quick Feature Branch

```bash
# 1. Open the launcher
fvdb-open

# 2. Select "n" for new worktree
# 3. Choose "1) Create new branch"
# 4. Enter branch name: my-feature

# 5. Cursor opens with the new worktree
```

## Directory Structure

When you create worktrees, they are placed alongside your main repository:

```
~/github/
├── fvdb-core/                    # Main clone (stays on main)
├── fvdb-core-my-feature/         # Worktree for my-feature branch
├── fvdb-core-issue-123/          # Worktree for issue #123
├── fvdb-reality-capture/         # Main clone
├── fvdb-reality-capture-issue-187/
└── fvdb-reality-capture-experiment/
```

## Tips

1. **Keep main clean**: Leave your main clones on the `main` branch. Use worktrees for all feature work.

2. **One issue per worktree**: Create a dedicated worktree for each issue you're working on.

3. **Clean up regularly**: Use `fvdb-close` to remove worktrees you're done with. This keeps your disk clean and git refs tidy.

4. **Use Claude Code for automation**: The `--claude` flag with `fvdb-issue` auto-starts Claude Code with the issue context, perfect for automated workflows.

5. **Visual differentiation**: Install the Peacock VS Code extension to color-code different worktree windows.

## Troubleshooting

### "FVDB_CORE_PATH is not set"

Create the config file:
```bash
cat > ~/.fvdb-devtools.conf << 'EOF'
FVDB_CORE_PATH=/your/path/to/fvdb-core
FVDB_RC_PATH=/your/path/to/fvdb-reality-capture
EOF
```

### "command not found: fvdb-open"

Ensure `~/bin` is in your PATH:
```bash
export PATH="$HOME/bin:$PATH"
```

Add this to your `~/.bashrc` or `~/.zshrc` to make it permanent.

### Worktree shows as "detached"

This happens when a worktree is created for a commit rather than a branch. To fix:
1. Close the worktree: `fvdb-close <name>`
2. Create a new one using `fvdb-open` and select "Use existing branch"

### "gh" command not found

Install the GitHub CLI: https://cli.github.com/

### Clipboard not working

On Linux without a display (SSH), clipboard operations won't work. The prompt will be printed to the terminal instead.

## Uninstalling

```bash
cd devtools/worktree
./install.sh --uninstall
```

Or manually:
```bash
rm ~/bin/fvdb-{open,issue,close}
rm ~/.fvdb-devtools.conf  # Optional
```
