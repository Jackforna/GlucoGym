import numpy as np
#from envJ import Gluco_env
from envG import Gluco_env2
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import ProgressBarCallback
import os
os.environ["TQDM_DISABLE_RICH"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def main():
    env = Gluco_env2()
    check_env(env, warn=True)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=100000, progress_bar=ProgressBarCallback())
    model.save("ppo_model")

    window_size = 100

    rewards, gluco_levels = env.get_res() #valori che ritorna l'environment

    rewards_smooth = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')

    fig = plt.figure(figsize=(14,10))
    fig.subplots_adjust(hspace=0.5)  # Aggiunge spazio tra i grafici
    plt.subplot(2,1,1)
    plt.plot(rewards, color='blue', alpha=0.15, linewidth=1, label="Rewards (raw)")
    plt.plot(rewards_smooth, color='blue', linewidth=2.5, label="Rewards (smoothed)")
    plt.title("Rewards")
    plt.xlabel("Time (steps)")
    plt.ylabel("Reward value")
    plt.grid(True)
    plt.legend()

    gluco_smooth = np.convolve(gluco_levels, np.ones(window_size)/window_size, mode='valid')

    # Curva glicemia
    plt.subplot(2,1,2)
    plt.plot(gluco_levels, color='blue', alpha=0.15, linewidth=1, label="Glucose (raw)")
    plt.plot(gluco_smooth, color='blue', linewidth=2.5, label="Glucose (smoothed)")
    plt.ylim(0, 400)  # Limiti asse Y
    plt.axhspan(70, 180, color="green", alpha=0.15) # Zona verde (70–180 mg/dL)
    plt.axhline(70, color="red", linestyle="--", linewidth=1)
    plt.axhline(250, color="red", linestyle="--", linewidth=1) # Linee rosse tratteggiate per ipo/iper
    plt.xlabel("Time (steps)")
    plt.ylabel("Glucose [mg/dL]")
    plt.title("Glucose Levels Over Time")
    plt.grid(True, alpha=0.3)
    plt.legend()


    plt.show()
    env.close()


if __name__ == "__main__":
    main()
