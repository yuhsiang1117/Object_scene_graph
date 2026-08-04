#!/bin/bash
set -e
echo "Installing Claude Code CLI"

# Install Claude Code CLI using official installer
curl -fsSL https://claude.ai/install.sh | bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

echo "Claude Code CLI installed successfully!"
echo "Version information:"
claude --version || true

echo "Claude Code installation completed!"