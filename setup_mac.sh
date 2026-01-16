#!/bin/bash
set -e

echo "Creating conda environment..."
conda create -n cleanrl python=3.9 -y

echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate cleanrl

echo "Installing packages..."
pip install "cleanrl[ppo_fix_continuous_action]"
pip install -U "wandb>=0.22.3"

if command -v brew &> /dev/null; then
    echo "Installing glfw..."
    brew install glfw
    export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH
    
    # Add to shell profile
    SHELL_PROFILE=""
    [ -f "$HOME/.zshrc" ] && SHELL_PROFILE="$HOME/.zshrc"
    [ -f "$HOME/.bash_profile" ] && SHELL_PROFILE="$HOME/.bash_profile"
    [ -f "$HOME/.bashrc" ] && SHELL_PROFILE="$HOME/.bashrc"
    
    if [ -n "$SHELL_PROFILE" ] && ! grep -q "DYLD_LIBRARY_PATH=/opt/homebrew/lib" "$SHELL_PROFILE"; then
        echo "" >> "$SHELL_PROFILE"
        echo "export DYLD_LIBRARY_PATH=/opt/homebrew/lib:\$DYLD_LIBRARY_PATH" >> "$SHELL_PROFILE"
    fi
fi

echo "Setup complete. Run: conda activate cleanrl"
