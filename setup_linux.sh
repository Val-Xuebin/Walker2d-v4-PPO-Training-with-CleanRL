#!/bin/bash
set -e

# echo "Creating conda environment..."
# conda create -n cleanrl python=3.9 -y

# echo "Activating environment..."
# eval "$(conda shell.bash hook)"
# conda activate cleanrl

echo "Installing packages..."
pip install "cleanrl[ppo_fix_continuous_action]"
pip install -U "wandb>=0.22.3"

if [ -z "$DISPLAY" ]; then
    echo "Setting MUJOCO_GL=osmesa for headless rendering..."
    export MUJOCO_GL=osmesa
    
    # Add to shell profile
    SHELL_PROFILE=""
    [ -f "$HOME/.bashrc" ] && SHELL_PROFILE="$HOME/.bashrc"
    [ -f "$HOME/.bash_profile" ] && SHELL_PROFILE="$HOME/.bash_profile"
    [ -f "$HOME/.profile" ] && SHELL_PROFILE="$HOME/.profile"
    
    if [ -n "$SHELL_PROFILE" ] && ! grep -q "MUJOCO_GL=osmesa" "$SHELL_PROFILE"; then
        echo "" >> "$SHELL_PROFILE"
        echo "export MUJOCO_GL=osmesa" >> "$SHELL_PROFILE"
    fi
fi

echo "Setup complete. Run: conda activate cleanrl"
