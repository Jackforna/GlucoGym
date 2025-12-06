import numpy as np
from envJ import Gluco_env
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

# Carica il modello salvato
model = PPO.load("ppo_model")

# Crea nuovo environment per test
env_test = Gluco_env()
obs, _ = env_test.reset()

print("\n--- Test di una giornata (24 step) ---")

glucose_history = []
hour_history = []

for t in range(24):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, trunc, info = env_test.step(action)

    gluco = obs[0]
    hour = int(obs[1])

    glucose_history.append(gluco)
    hour_history.append(hour)

    print(f"Step {t:2d} | Hour: {hour:2d} | Glucose: {gluco:6.1f} mg/dL | Reward: {reward:.3f} | Action: {action}")

print("\nTest completato.")