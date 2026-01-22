"""Training configuration for PPO Walker2d-v4"""
import torch

class Config:
    # Experiment settings
    exp_name = "ppo_fix_continuous_action"
    seed = 2
    torch_deterministic = True
    cuda = True
    
    # Logging
    track = True
    wandb_project_name = "cleanRL"
    wandb_entity = None
    capture_video = False
    
    # Model saving
    save_model = True
    upload_model = False
    hf_entity = "sdpkjc"
    checkpoint_step = 100000  # Save checkpoint every N steps
    
    # Environment
    env_id = "Walker2d-v4"
    total_timesteps = 1000000
    num_envs = 1
    
    # Agent network architecture
    agent_network = "mlp"  # "mlp", "lstm", or "attn"
    
    # Algorithm hyperparameters
    learning_rate = 3e-4
    num_steps = 2048
    anneal_lr = True
    gamma = 0.99
    gae_lambda = 0.95
    num_minibatches = 32
    update_epochs = 10
    norm_adv = True
    clip_coef = 0.2
    clip_vloss = True
    ent_coef = 0.0
    vf_coef = 0.5
    max_grad_norm = 0.5
    target_kl = None
    
    # Device
    @property
    def device(self):
        """Get device (cuda if available and cuda=True, else cpu)"""
        return torch.device("cuda" if torch.cuda.is_available() and self.cuda else "cpu")

