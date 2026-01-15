#!/usr/bin/env python3
"""Unified evaluation script for PPO models"""

import argparse
import os
import glob
import torch
import gymnasium as gym
from ppo_fix_continuous_action import Agent, make_eval_env
from huggingface_hub import hf_hub_download

def evaluate_model(model_path, env_id="Walker2d-v4", eval_episodes=10, capture_video=False, run_name=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if run_name is None:
        run_name = f"eval_{os.path.basename(model_path).replace('.cleanrl_model', '')}"
    
    # Create temp env to get obs_rms structure
    temp_envs = gym.vector.SyncVectorEnv([make_eval_env(env_id, 0, False, "temp", None)])
    agent = Agent(temp_envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    
    # Get obs_rms from loaded agent
    obs_rms = agent.obs_rms
    
    # Create eval env with obs_rms
    envs = gym.vector.SyncVectorEnv([make_eval_env(env_id, 0, capture_video, run_name, obs_rms)])
    
    obs, _ = envs.reset()
    episodic_returns, episodic_lengths = [], []
    
    print(f"\nEvaluating: {model_path}")
    print(f"Episodes: {eval_episodes}, Video: {capture_video}")
    
    while len(episodic_returns) < eval_episodes:
        actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
        next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
        
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                ret = float(info['episode']['r'])
                length = int(info['episode']['l'])
                print(f"  episode={len(episodic_returns)}, return={ret:.2f}, length={length}")
                episodic_returns += [ret]
                episodic_lengths += [length]
        obs = next_obs
    
    envs.close()
    
    mean_return = sum(episodic_returns) / len(episodic_returns)
    std_return = torch.std(torch.tensor(episodic_returns)).item()
    
    print(f"  Mean: {mean_return:.2f} ± {std_return:.2f}, Range: [{min(episodic_returns):.2f}, {max(episodic_returns):.2f}]")
    
    if capture_video:
        print(f"  Video saved to: videos/{run_name}/")
    
    return episodic_returns, episodic_lengths

def find_models(pattern=None):
    """Find model files matching pattern"""
    if pattern:
        models = glob.glob(pattern)
    else:
        models = glob.glob("runs/**/*.cleanrl_model", recursive=True)
    
    models = [m for m in models if os.path.isfile(m)]
    return sorted(models)

def load_model_from_hf(hf_repo, exp_name="ppo_fix_continuous_action"):
    """Load model from Hugging Face Hub"""
    try:
        model_path = hf_hub_download(repo_id=hf_repo, filename=f"{exp_name}.cleanrl_model")
        return model_path
    except Exception as e:
        print(f"Error loading from Hugging Face {hf_repo}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate PPO models")
    parser.add_argument("--model", type=str, nargs="+", help="Model path(s), pattern(s), or HF repo(s)")
    parser.add_argument("--hf-repo", type=str, help="Hugging Face repo (e.g., sdpkjc/Walker2d-v4-ppo_fix_continuous_action-seed4)")
    parser.add_argument("--env-id", type=str, default="Walker2d-v4", help="Environment ID")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--video", action="store_true", help="Capture video")
    parser.add_argument("--list", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    if args.list:
        models = find_models()
        print(f"Found {len(models)} local models:")
        for i, m in enumerate(models, 1):
            size = os.path.getsize(m) / 1024
            print(f"  {i}. {m} ({size:.1f}KB)")
        return
    
    all_models = []
    
    # Handle Hugging Face repo
    if args.hf_repo:
        model_path = load_model_from_hf(args.hf_repo)
        if model_path:
            all_models.append(model_path)
            print(f"Loaded from Hugging Face: {args.hf_repo}")
    
    # Handle --model arguments
    if args.model:
        for pattern in args.model:
            # Check if it's a Hugging Face repo format (contains /)
            if "/" in pattern and not os.path.exists(pattern) and not "*" in pattern:
                model_path = load_model_from_hf(pattern)
                if model_path:
                    all_models.append(model_path)
            elif "*" in pattern or "?" in pattern:
                all_models.extend(find_models(pattern))
            elif os.path.isfile(pattern):
                all_models.append(pattern)
            else:
                print(f"Warning: Model not found: {pattern}")
    
    # Default: use final model from latest run
    if not all_models:
        runs = glob.glob("runs/*/ppo_fix_continuous_action.cleanrl_model")
        if runs:
            all_models = [max(runs, key=os.path.getmtime)]
            print(f"Using latest final model: {all_models[0]}")
        else:
            print("No models found. Use --list to see available models or --hf-repo to load from Hugging Face.")
            return
    
    if not all_models:
        print("No valid models found.")
        return
    
    print(f"\nEvaluating {len(all_models)} model(s)...")
    print("="*60)
    
    results = []
    for model_path in all_models:
        try:
            returns, lengths = evaluate_model(
                model_path,
                env_id=args.env_id,
                eval_episodes=args.episodes,
                capture_video=args.video
            )
            results.append({
                "model": model_path,
                "mean": sum(returns) / len(returns),
                "std": torch.std(torch.tensor(returns)).item(),
                "min": min(returns),
                "max": max(returns)
            })
        except Exception as e:
            print(f"Error evaluating {model_path}: {e}")
    
    if len(results) > 1:
        print("\n" + "="*60)
        print("Summary:")
        print("="*60)
        for r in results:
            print(f"{os.path.basename(r['model']):50s} Mean: {r['mean']:7.2f} ± {r['std']:6.2f}")

if __name__ == "__main__":
    main()

