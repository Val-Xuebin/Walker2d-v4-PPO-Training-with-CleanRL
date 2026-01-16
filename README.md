# Walker2d-v4 PPO Training

PPO training for Walker2d-v4 using CleanRL framework.

## Setup

### Quick Setup (Recommended)

Run the appropriate setup script for your platform:

**macOS:**

```bash
./setup_mac.sh
```

**Linux:**

```bash
./setup_linux.sh
```



Key Versions

- Gym: 1.1.1
- Wandb: 0.24.0

### Manual Setup

Alternatively, you can set up manually:

```bash
conda create -n cleanrl python=3.9 -y
conda activate cleanrl
pip install "cleanrl[ppo_fix_continuous_action]"
pip install -U "wandb>=0.22.3"
```

**Rendering backend for video generation**:

- **macOS**: `brew install glfw` and `export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH`
- **Linux (headless)**: `export MUJOCO_GL=osmesa` or `export MUJOCO_GL=egl`

## Configuration

Edit `config.py` for training parameters. Main settings:

- `total_timesteps`: training steps (default 1000000)
- `seed`: random seed
- `env_id`: environment name
- Algorithm hyperparameters

## Training

```bash
python ppo_fix_continuous_action.py
```

Checkpoints are saved every `checkpoint_step` steps (default: 100k). Final model saved as `ppo_fix_continuous_action.cleanrl_model`.

Training runs for 1 million timesteps by default. Progress logged to tensorboard and wandb if enabled.

## Evaluation

```bash
# List available models
python eval.py --list

# Evaluate single model
python eval.py --model runs/*/ppo_fix_continuous_action.cleanrl_model --episodes 10

# Evaluate multiple models
python eval.py --model runs/*/ppo_fix_continuous_action-100000.cleanrl_model runs/*/ppo_fix_continuous_action-500000.cleanrl_model

# Evaluate with video
python eval.py --model runs/*/ppo_fix_continuous_action.cleanrl_model --episodes 1 --video

# Evaluate from Hugging Face
python eval.py --hf-repo sdpkjc/Walker2d-v4-ppo_fix_continuous_action-seed4 --episodes 10
```

**Video Generation**: Videos are saved to `videos/{run_name}/` directory. Ensure rendering backend is configured (see Setup section).

## Implementation Notes

Key implementation details:

1. **Gymnasium API compatibility**

   - `TransformObservation` wrapper requires 3 args: `(env, func, observation_space)`
   - Fixed in `make_env` and `make_eval_env` functions
2. **Vectorized environment episode statistics**

   - Updated logging logic to support both vectorized (`episode` key) and non-vectorized (`final_info` key) environments
   - Ensures correct `episodic_return` extraction in all cases
3. **Rendering backend**

   - Automatic backend selection for headless environments (OSMesa/EGL)
   - Manual override via `MUJOCO_GL` environment variable

## Files

- `ppo_fix_continuous_action.py`: training script
- `config.py`: configuration
- `eval.py`: unified evaluation script (supports multiple models, metrics, video)
- Models: `runs/{run_name}/ppo_fix_continuous_action-{step}.cleanrl_model` (~49KB each)
