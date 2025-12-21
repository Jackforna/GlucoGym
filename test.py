import numpy as np
from envG import Gluco_env2
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

# Carica il modello salvato
model = PPO.load("ppo_model")

# Crea nuovo environment per test
env_test = Gluco_env2()
obs, _ = env_test.reset()

print("\n--- Test di una giornata (24 step) ---")

glucose_history = []
hour_history = []
prec_hour = -1
i = 0
ranges = [0, 0, 0, 0, 0]

for t in range(720):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, trunc, info = env_test.step(action)

    gluco = info[0]
    #minutes = t
    #hour = int(minutes / 60)
    #minutes -= hour * 60
    hour = t
    carbo = info[2]
    insulin = info[1]

    if gluco < 55:
        ranges[0] += 1
    elif gluco < 70:
        ranges[1] += 1
    elif gluco < 180:
        ranges[2] += 1
    elif gluco < 220:
        ranges[3] += 1
    else:
        ranges[4] += 1

    glucose_history.append(gluco)
    hour_history.append(hour)

    #print(f"Step {t:2d} | Hour: {hour:2d}:00 | Glucose: {gluco:6.1f} mg/dL | Carbo: {carbo:6.1f} g | Insulin {insulin:6.1f} | Reward: {reward:.3f} | Action: {action}")

fig1 = plt.figure(figsize=(14,10))
plt.plot(glucose_history, color='blue', linewidth=1, label="Glucose")
plt.ylim(0, 400)  # Limiti asse Y
plt.axhspan(70, 180, color="green", alpha=0.15) # Zona verde (70–180 mg/dL)
plt.axhline(70, color="red", linestyle="--", linewidth=1)
plt.axhline(250, color="red", linestyle="--", linewidth=1) # Linee rosse tratteggiate per ipo/iper
plt.xlabel("Time (hours)")
plt.ylabel("Glucose [mg/dL]")
plt.title("Glucose Levels Over Time")
plt.grid(True, alpha=0.3)
plt.legend()

labels = [
        'Very Low\n(<55)', 
        'Low\n(55-70)', 
        'In Range\n(70-180)', 
        'High\n(180-220)', 
        'Very High\n(>220)'
    ]
    
    # Calcolo delle percentuali
total_steps = sum(ranges)
percentages = [(val / total_steps) * 100 for val in ranges]
    
    # Colori standard per la gestione del diabete
colors = ['#8b0000', '#ff4500', '#2ecc71', '#f39c12', '#e74c3c']

fig2 = plt.figure(figsize=(10, 6))
bars = plt.bar(labels, percentages, color=colors, edgecolor='black', alpha=0.8)

    # Aggiunta delle percentuali scritte sopra le barre
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.title('Time in Range (TIR) - Analisi Mensile', fontsize=14)
plt.ylabel('Percentuale del tempo (%)')
plt.ylim(0, max(percentages) + 15) # Spazio extra per le etichette
plt.grid(axis='y', linestyle='--', alpha=0.7)
    
plt.tight_layout()


plt.show()

print("\nTest completato.")