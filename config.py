"""Training configuration for PPO Walker2d-v4"""

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
    
    # Environment
    env_id = "Walker2d-v4"
    total_timesteps = 1000000
    num_envs = 1
    
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
    
    # Computed values
    @property
    def batch_size(self):
        return int(self.num_envs * self.num_steps)
    
    @property
    def minibatch_size(self):
        return int(self.batch_size // self.num_minibatches)
    
    @property
    def num_checkpoints(self):
        """Number of model checkpoints to save during training"""
        return 10
    
    @property
    def checkpoint_interval(self):
        """Steps between checkpoints"""
        return self.total_timesteps // self.num_checkpoints

