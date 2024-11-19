import gymnasium as gym
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy
import os 

# Make environment
env = gym.make("BipedalWalker-v3", render_mode="human")
observation, info = env.reset()

# load the winner
local_dir = os.path.dirname(__file__)
results_dir = os.path.join(local_dir, 'results')
model = DDPG.load(os.path.join(local_dir, results_dir, 'ddpg_BipedalWalker'), env=env)

# Evaluate
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)


# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")