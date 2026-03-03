#!/bin/bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# Install fvdb worktree tools
# Usage: ./install.sh [--uninstall]
#
# This script:
# 1. Copies fvdb-open, fvdb-issue, fvdb-close to ~/bin
# 2. Makes them executable
# 3. Adds ~/bin to PATH if not present
# 4. Creates a config file template if needed

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$HOME/bin"
CONFIG_FILE="$HOME/.fvdb-devtools.conf"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

TOOLS=(fvdb-open fvdb-issue fvdb-close)

uninstall() {
    echo -e "${BLUE}Uninstalling fvdb worktree tools...${NC}"

    for tool in "${TOOLS[@]}"; do
        if [[ -f "${INSTALL_DIR}/${tool}" ]]; then
            rm "${INSTALL_DIR}/${tool}"
            echo -e "  Removed ${tool}"
        fi
    done

    echo ""
    echo -e "${GREEN}Uninstalled.${NC}"
    echo ""
    echo "Note: ~/.fvdb-devtools.conf was not removed."
    echo "Delete it manually if no longer needed."
    exit 0
}

# Parse arguments
if [[ "$1" == "--uninstall" || "$1" == "-u" ]]; then
    uninstall
fi

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "Usage: ./install.sh [--uninstall]"
    echo ""
    echo "Options:"
    echo "  --uninstall, -u   Remove installed tools"
    echo "  --help, -h        Show this help"
    exit 0
fi

echo -e "${BLUE}Installing fvdb worktree tools...${NC}"
echo ""

# Create ~/bin if it doesn't exist
if [[ ! -d "$INSTALL_DIR" ]]; then
    echo "Creating ${INSTALL_DIR}..."
    mkdir -p "$INSTALL_DIR"
fi

# Copy tools
for tool in "${TOOLS[@]}"; do
    if [[ ! -f "${SCRIPT_DIR}/${tool}" ]]; then
        echo -e "${RED}Error: ${tool} not found in ${SCRIPT_DIR}${NC}"
        exit 1
    fi

    cp "${SCRIPT_DIR}/${tool}" "${INSTALL_DIR}/${tool}"
    chmod +x "${INSTALL_DIR}/${tool}"
    echo -e "  ${GREEN}✓${NC} Installed ${tool}"
done

echo ""

# Check if ~/bin is in PATH
if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
    echo -e "${YELLOW}~/bin is not in your PATH.${NC}"
    echo ""

    # Detect shell config file, preferring the active shell
    SHELL_CONFIG=""

    # Determine current shell name from $SHELL, falling back to ps if needed
    SHELL_NAME="${SHELL##*/}"
    if [[ -z "$SHELL_NAME" ]]; then
        if command -v ps >/dev/null 2>&1; then
            SHELL_NAME="$(ps -p $$ -o comm= 2>/dev/null)"
        fi
    fi

    case "$SHELL_NAME" in
        bash)
            if [[ -f "$HOME/.bashrc" ]]; then
                SHELL_CONFIG="$HOME/.bashrc"
            elif [[ -f "$HOME/.bash_profile" ]]; then
                SHELL_CONFIG="$HOME/.bash_profile"
            fi
            ;;
        zsh)
            if [[ -f "$HOME/.zshrc" ]]; then
                SHELL_CONFIG="$HOME/.zshrc"
            fi
            ;;
    esac

    # Fallback: pick a reasonable config file based on existence only
    if [[ -z "$SHELL_CONFIG" ]]; then
        if [[ -f "$HOME/.bashrc" ]]; then
            SHELL_CONFIG="$HOME/.bashrc"
        elif [[ -f "$HOME/.bash_profile" ]]; then
            SHELL_CONFIG="$HOME/.bash_profile"
        elif [[ -f "$HOME/.zshrc" ]]; then
            SHELL_CONFIG="$HOME/.zshrc"
        fi
    fi

    if [[ -n "$SHELL_CONFIG" ]]; then
        echo "Add this line to $SHELL_CONFIG:"
        echo ""
        echo "  export PATH=\"\$HOME/bin:\$PATH\""
        echo ""
        read -p "Add it now? [y/n]: " add_path
        if [[ "$add_path" == "y" || "$add_path" == "Y" ]]; then
            echo '' >> "$SHELL_CONFIG"
            echo '# fvdb devtools' >> "$SHELL_CONFIG"
            echo 'export PATH="$HOME/bin:$PATH"' >> "$SHELL_CONFIG"
            echo -e "${GREEN}Added to ${SHELL_CONFIG}${NC}"
            echo "Run 'source ${SHELL_CONFIG}' or start a new terminal to use the tools."
        fi
    else
        echo "Add this to your shell configuration:"
        echo "  export PATH=\"\$HOME/bin:\$PATH\""
    fi
    echo ""
fi

# Check/create config file
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo -e "${YELLOW}Configuration file not found.${NC}"
    echo ""
    echo "The tools need to know where your fvdb repositories are located."
    echo ""

    read -p "Create configuration file now? [y/n]: " create_config
    if [[ "$create_config" == "y" || "$create_config" == "Y" ]]; then
        echo ""

        # Try to auto-detect paths
        DEFAULT_CORE=""
        DEFAULT_RC=""

        # Look for common locations
        for dir in "$HOME/github" "$HOME/repos" "$HOME/code" "$HOME/src" "$HOME/projects"; do
            if [[ -d "${dir}/fvdb-core" ]]; then
                DEFAULT_CORE="${dir}/fvdb-core"
            elif [[ -d "${dir}/fvdb" ]]; then
                DEFAULT_CORE="${dir}/fvdb"
            fi
            if [[ -d "${dir}/fvdb-reality-capture" ]]; then
                DEFAULT_RC="${dir}/fvdb-reality-capture"
            fi
        done

        # Prompt for paths
        if [[ -n "$DEFAULT_CORE" ]]; then
            read -p "fvdb-core path [$DEFAULT_CORE]: " CORE_PATH
            CORE_PATH="${CORE_PATH:-$DEFAULT_CORE}"
        else
            read -p "fvdb-core path: " CORE_PATH
        fi

        if [[ -n "$DEFAULT_RC" ]]; then
            read -p "fvdb-reality-capture path [$DEFAULT_RC]: " RC_PATH
            RC_PATH="${RC_PATH:-$DEFAULT_RC}"
        else
            read -p "fvdb-reality-capture path: " RC_PATH
        fi

        # Expand ~ in paths
        CORE_PATH="${CORE_PATH/#\~/$HOME}"
        RC_PATH="${RC_PATH/#\~/$HOME}"

        # Validate paths
        if [[ ! -d "$CORE_PATH" ]]; then
            echo -e "${YELLOW}Warning: $CORE_PATH does not exist${NC}"
        fi
        if [[ ! -d "$RC_PATH" ]]; then
            echo -e "${YELLOW}Warning: $RC_PATH does not exist${NC}"
        fi

        # Write config file
        cat > "$CONFIG_FILE" << EOF
# fvdb devtools configuration
# Edit these paths to match your local setup

# Required: Path to your main fvdb-core clone
FVDB_CORE_PATH="$CORE_PATH"

# Required: Path to your main fvdb-reality-capture clone
FVDB_RC_PATH="$RC_PATH"

# Optional: GitHub repository names (defaults shown)
# FVDB_CORE_GH_REPO="openvdb/fvdb"
# FVDB_RC_GH_REPO="openvdb/fvdb-reality-capture"
EOF

        echo -e "${GREEN}Created ${CONFIG_FILE}${NC}"
    else
        echo "Create the config file manually:"
        echo ""
        echo "  cat > ~/.fvdb-devtools.conf << 'EOF'"
        echo "  FVDB_CORE_PATH=/path/to/fvdb-core"
        echo "  FVDB_RC_PATH=/path/to/fvdb-reality-capture"
        echo "  EOF"
    fi
    echo ""
else
    echo -e "${GREEN}Configuration file exists: ${CONFIG_FILE}${NC}"
fi

echo ""
echo -e "${GREEN}Installation complete!${NC}"
echo ""
echo "Available commands:"
echo "  fvdb-open   - Open worktrees in Cursor"
echo "  fvdb-issue  - Create worktree for a GitHub issue"
echo "  fvdb-close  - Remove a worktree"
echo ""
echo "Run 'fvdb-open --help' for usage information."
