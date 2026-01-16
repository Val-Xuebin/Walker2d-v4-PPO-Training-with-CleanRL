# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import copy
import os
import random
import time
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from config import Config


# https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/wrappers/normalize.py
class RunningMeanStd(nn.Module):
    def __init__(self, epsilon=1e-4, shape=()):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float64))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(epsilon, dtype=torch.float64))

    def update(self, x):
        x = torch.as_tensor(x, dtype=torch.float64).to(self.mean.device)
        batch_mean = torch.mean(x, dim=0).to(self.mean.device)
        batch_var = torch.var(x, dim=0, unbiased=False).to(self.mean.device)
        batch_count = x.shape[0]

        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class NormalizeObservation(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        gym.utils.RecordConstructorArgs.__init__(self, epsilon=epsilon)
        gym.Wrapper.__init__(self, env)

        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False

        if self.is_vector_env:
            self.obs_rms = RunningMeanStd(shape=self.single_observation_space.shape)
        else:
            self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.epsilon = epsilon

        self.enable = True
        self.freeze = False

    def step(self, action):
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        if self.is_vector_env:
            obs = self.normalize(obs)
        else:
            obs = self.normalize(np.array([obs]))[0]
        return obs, rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        if self.is_vector_env:
            return self.normalize(obs), info
        else:
            return self.normalize(np.array([obs]))[0], info

    def normalize(self, obs):
        if not self.freeze:
            self.obs_rms.update(obs)
        if self.enable:
            return (obs - self.obs_rms.mean.cpu().numpy()) / np.sqrt(self.obs_rms.var.cpu().numpy() + self.epsilon)
        return obs


class NormalizeReward(gym.core.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        gym.utils.RecordConstructorArgs.__init__(self, gamma=gamma, epsilon=epsilon)
        gym.Wrapper.__init__(self, env)

        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False

        self.return_rms = RunningMeanStd(shape=())
        self.returns = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

        self.enable = True
        self.freeze = False

    def step(self, action):
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        if not self.is_vector_env:
            rews = np.array([rews])
        self.returns = self.returns * self.gamma * (1 - terminateds) + rews
        rews = self.normalize(rews)
        if not self.is_vector_env:
            rews = rews[0]
        return obs, rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        # self.returns = np.zeros(self.num_envs)
        return self.env.reset(**kwargs)

    def normalize(self, rews):
        if not self.freeze:
            self.return_rms.update(self.returns)
        if self.enable:
            return rews / np.sqrt(self.return_rms.var.cpu().numpy() + self.epsilon)
        return rews

    def get_returns(self):
        return self.returns


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
):
    # Setup rendering backend for headless environments when capturing video
    if capture_video and os.environ.get("DISPLAY") is None:
        if os.environ.get("MUJOCO_GL") is None:
            import ctypes.util
            osmesa_path = ctypes.util.find_library('OSMesa')
            if osmesa_path:
                os.environ["MUJOCO_GL"] = "osmesa"
            else:
                os.environ["MUJOCO_GL"] = "egl"
    
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, capture_video, run_name)])
    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, capture_video, run_name, agent.obs_rms)])

    obs, _ = envs.reset()
    episodic_returns, episodic_lengths = [], []
    while len(episodic_returns) < eval_episodes:
        actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
        next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
        # Support both vectorized (episode key) and non-vectorized (final_info key) environments
        if "episode" in infos:
            episode_info = infos["episode"]
            if episode_info.get("_r", [False])[0]:  # Check if there's a valid episode
                episodic_return = episode_info["r"][0]
                episodic_length = episode_info["l"][0]
                print(f"eval_episode={len(episodic_returns)}, episodic_return={episodic_return}")
                episodic_returns += [episodic_return]
                episodic_lengths += [episodic_length]
        elif "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
                episodic_lengths += [info["episode"]["l"]]
        obs = next_obs

    return episodic_returns, episodic_lengths


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        env = NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
        env = NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def make_eval_env(env_id, idx, capture_video, run_name, obs_rms=None):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        env = NormalizeObservation(env)
        if obs_rms is not None:
            env.obs_rms = copy.deepcopy(obs_rms)
        env.freeze = True
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
        return env

    return thunk


def get_rms(env):
    obs_rms, return_rms = None, None
    env_point = env
    while hasattr(env_point, "env"):
        if isinstance(env_point, NormalizeObservation):
            obs_rms = copy.deepcopy(env_point.obs_rms)
            break
        env_point = env_point.env
    else:
        raise RuntimeError("can't find NormalizeObservation")

    env_point = env
    while hasattr(env_point, "env"):
        if isinstance(env_point, NormalizeReward):
            return_rms = copy.deepcopy(env_point.return_rms)
            break
        env_point = env_point.env
    else:
        raise RuntimeError("can't find NormalizeReward")

    return obs_rms, return_rms


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
        self.obs_rms = RunningMeanStd(shape=envs.single_observation_space.shape)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


if __name__ == "__main__":
    config = Config()
    run_name = f"{config.env_id}__{config.exp_name}__{config.seed}__{int(time.time())}"
    if config.track:
        import wandb

        wandb.init(
            project=config.wandb_project_name,
            entity=config.wandb_entity,
            sync_tensorboard=True,
            config={k: v for k, v in vars(config).items() if not k.startswith('_')},
            name=run_name,
            monitor_gym=False,  # Disabled due to incompatibility with gymnasium >= 1.0
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    config_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in config_dict.items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic

    # Computed values
    device = config.device
    batch_size = int(config.num_envs * config.num_steps)
    minibatch_size = int(batch_size // config.num_minibatches)
    # Calculate checkpoint steps based on config
    # Generate checkpoints every checkpoint_step until total_timesteps
    checkpoint_steps = []
    step = config.checkpoint_step
    while step <= config.total_timesteps:
        checkpoint_steps.append(step)
        step += config.checkpoint_step

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(config.env_id, i, config.capture_video, run_name, config.gamma) for i in range(config.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((config.num_steps, config.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((config.num_steps, config.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((config.num_steps, config.num_envs)).to(device)
    rewards = torch.zeros((config.num_steps, config.num_envs)).to(device)
    dones = torch.zeros((config.num_steps, config.num_envs)).to(device)
    values = torch.zeros((config.num_steps, config.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=config.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(config.num_envs).to(device)
    num_updates = config.total_timesteps // batch_size

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if config.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * config.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, config.num_steps):
            global_step += 1 * config.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            # https://github.com/DLR-RM/stable-baselines3/pull/658
            for idx, trunc in enumerate(truncations):
                if trunc and not terminations[idx]:
                    real_next_obs = infos["final_observation"][idx]
                    with torch.no_grad():
                        terminal_value = agent.get_value(torch.Tensor(real_next_obs).to(device)).reshape(1, -1)[0][0]
                    rewards[step][idx] += config.gamma * terminal_value

            # Save checkpoint at specified intervals
            if global_step in checkpoint_steps:
                obs_rms, return_rms = get_rms(envs.envs[0])
                agent.obs_rms = copy.deepcopy(get_rms(envs.envs[0])[0])
                model_path = f"runs/{run_name}/{config.exp_name}-{global_step}.cleanrl_model"
                torch.save(agent.state_dict(), model_path)
                print(f"model saved to {model_path}")

                episodic_returns, episodic_lengths = evaluate(
                    model_path,
                    make_eval_env,
                    config.env_id,
                    eval_episodes=3,
                    run_name=f"{run_name}-eval",
                    Model=Agent,
                    device=device,
                    capture_video=False,
                )

                print(episodic_returns, episodic_lengths)
                writer.add_scalar("charts/eval/episodic_return", np.mean(episodic_returns), global_step)
                writer.add_scalar("charts/eval/episodic_length", np.mean(episodic_lengths), global_step)

            # Only print when at least 1 env is done
            # Support both vectorized (episode key) and non-vectorized (final_info key) environments
            if "episode" in infos:
                # Vectorized environment: episode info is directly in infos
                episode_info = infos["episode"]
                if episode_info.get("_r", [False])[0]:  # Check if there's a valid episode
                    episodic_return = episode_info["r"][0]
                    episodic_length = episode_info["l"][0]
                    print(f"global_step={global_step}, episodic_return={episodic_return}")
                    writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                    writer.add_scalar("charts/episodic_length", episodic_length, global_step)
            elif "final_info" in infos:
                # Non-vectorized environment: episode info is in final_info
                for info in infos["final_info"]:
                    # Skip the envs that are not done
                    if info is None:
                        continue
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(config.num_steps)):
                if t == config.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + config.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > config.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if config.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -config.clip_coef,
                        config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                optimizer.step()

            if config.target_kl is not None:
                if approx_kl > config.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if config.save_model:
        agent.obs_rms = copy.deepcopy(get_rms(envs.envs[0])[0])
        model_path = f"runs/{run_name}/{config.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

        episodic_returns, episodic_lengths = evaluate(
            model_path,
            make_eval_env,
            config.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=Agent,
            device=device,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if config.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{config.env_id}-{config.exp_name}-seed{config.seed}"
            repo_id = f"{config.hf_entity}/{repo_name}" if config.hf_entity else repo_name
            push_to_hub(config, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
