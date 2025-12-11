import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
import random

class Gluco_env2(gym.Env):

    def __init__(self):
        super(Gluco_env2, self).__init__()

        self.action_space = spaces.MultiDiscrete([24,5]) 

        # Ho alzato leggermente i limiti 'high' dello space per evitare crash tecnici di Gym
        # se la glicemia o la resistenza superano di poco i valori previsti.
        # (Non influisce sulla logica fisica, solo sulla validazione).
        self.observation_space = spaces.Box(
            low = np.array([0,0,0,0,0,0,0,0,0,10], dtype=np.float32),
            high = np.array([600,24,300,10,50,10,100,3,24,1000], dtype=np.float32),
            dtype = np.float32
        )

        self.state = np.array([100, 0, 0, 0, 0, 0, 6, 0, 0, 50], dtype=np.float32)
        
        self.rew_arr = []
        self.gluco_arr = []
        self.last_5_glucolevels = deque(maxlen=5) 
        self.day_insulin = []
        
        # Variabile per calcolare la stabilità (delta)
        self.last_glucose = 100.0
    
    def step(self, action):
        take_insulin, take_sugar = action

        if not self.filter_invalid_actions(action):
            # Penalità per azioni invalide, ma salviamo comunque i dati per coerenza
            self.rew_arr.append(-100.0)
            self.gluco_arr.append(self.state[0])
            return self.state, -100.0, False, False, {}

        gluco_level, hour, carbo, carbo_time, insulin, time_insulin, basal, sport, sport_time, insulin_resistance = self.state

        # Casting a int
        hour = int(hour)
        carbo_time = int(carbo_time)
        time_insulin = int(time_insulin)

        if take_insulin != 0:
            time_insulin = 1
            insulin = take_insulin*0.5
            self.day_insulin.append(insulin)
        
        if take_sugar != 0:
            gluco_level += 36 * take_sugar

        # --- LOGICA PASTI (Tua originale) ---
        if hour>=7 and hour<=9 and carbo_time==0: 
            if random.random()>0.4: carbo_time, carbo = 1, random.uniform(40,80)
        if hour>11 and hour<15 and carbo_time==0: 
            if random.random()>0.4: carbo_time, carbo = 1, random.uniform(60,100)
        if hour==14 and carbo_time==0: 
            carbo_time, carbo = 1, random.uniform(60,100)
        if hour>19 and hour<23 and carbo_time==0: 
            if random.random()>0.4: carbo_time, carbo = 1, random.uniform(40,100)
        if hour==22 and carbo_time==0: 
            carbo_time, carbo = 1, random.uniform(40,100)

        # --- FISICA CARBO (Tua originale) ---
        if carbo > 0: 
            if carbo_time == 1: gluco_level += 0.1 * carbo * 4
            elif carbo_time == 2: gluco_level += 0.35 * carbo * 4
            elif carbo_time == 3: gluco_level += 0.35 * carbo * 4
            elif carbo_time == 4: gluco_level += 0.2 * carbo * 4
            carbo_time += 1
            if carbo_time > 4:
                carbo = 0
                carbo_time = 0

        # Resistenza Circadiana
        circadian = [1.2, 1.1, 1.0, 0.9, 0.85, 0.9, 1.0, 1.1, 1.2, 1.25,
                     1.2, 1.1, 1.0, 0.95, 1.0, 1.1, 1.15, 1.2, 1.15, 1.1,
                     1.0, 0.95, 0.9, 1.0]
        safe_hour = hour % 24
        insulin_resistance = insulin_resistance * circadian[safe_hour]

        # --- FISICA INSULINA (Tua originale) ---
        if time_insulin > 0:
            insulin_profile = [0.2, 0.35, 0.30, 0.15] 
            if 1 <= time_insulin <= 4:
                idx = int(time_insulin) - 1
                gluco_level -= insulin * insulin_resistance * insulin_profile[idx]
            time_insulin += 1
            if time_insulin > 4:
                insulin = 0
                time_insulin = 0

        # --- MODIFICA 1: RUMORE RIDOTTO ---
        # Da +/- 10 a +/- 2. Fondamentale per la stabilità.
        gluco_level += random.uniform(-2, 2) 

        hour = (hour+1)%24

        # Logiche ricorrenti
        if hour == 6:
            self.last_5_glucolevels.append(gluco_level)
            if len(self.last_5_glucolevels) == 5:
                avg_morning = np.mean(self.last_5_glucolevels)
                if avg_morning < 90: basal -= 2
                elif avg_morning > 140: basal += 2
                basal = np.clip(basal, 5, 40)
                self.last_5_glucolevels.clear()

        if hour == 0:
            total_ins = sum(self.day_insulin) + basal
            if total_ins > 0:
                insulin_resistance = 1800.0/total_ins
            else:
                insulin_resistance = 50.0
            self.day_insulin.clear()

        basal_effect = basal * insulin_resistance * 0.01 
        gluco_level -= basal_effect

        # Clipping originale come richiesto (0, 400)
        gluco_level = np.clip(gluco_level, 0, 400)

        # --- CALCOLO REWARD ---
        # Gaussiana più larga (diviso 30 invece di 20) per "attrarre" l'agente anche quando è lontano
        reward = np.exp(-0.5 * ((gluco_level - 110) / 30) ** 2) * 2.0

        # Penalità base
        reward -= max(0, (70 - gluco_level) / 20)
        reward -= max(0, (gluco_level - 180) / 50)
        
        # MODIFICA 2: Tolto il blocco "if < 110 punish insulin". 
        # Puniamo solo se è < 70, per permettere correzioni anticipate.
        if gluco_level < 70 and take_insulin > 0:
            reward -= 15

        if 90 <= gluco_level <= 140:
            reward += 3 # Bonus originale
        elif 70 <= gluco_level <= 89 or 141 <= gluco_level <= 180:
            reward += 1 # Bonus ridotto per "zona ok"
        
        # Penalità forti
        if gluco_level < 60: reward -= 10 # Anticipiamo la penalità forte prima del "muro" a 50
        if gluco_level < 50: reward -= 50 # Muro (era 20, alzato per dire "qui non ci devi proprio andare")
        if gluco_level > 250: reward -= 5 # Iper grave
        
        # Penalità Iper aumentata progressivamente
        if gluco_level > 180:
             reward -= (gluco_level - 180) * 0.1

        # --- MODIFICA 3: STABILITÀ (DELTA) ---
        # Penalizziamo le variazioni brusche tra uno step e l'altro
        delta = abs(gluco_level - self.last_glucose)
        reward -= delta * 0.1 
        
        self.last_glucose = gluco_level

        # Logging
        self.rew_arr.append(reward)
        self.gluco_arr.append(gluco_level)
        
        self.state = np.array([gluco_level, hour, carbo, carbo_time, insulin, time_insulin, basal, sport, sport_time, insulin_resistance], dtype=np.float32)
        
        done = False
        truncated = False
        
        return self.state, float(reward), done, truncated, {}
    
    def filter_invalid_actions(self, action):
        take_insulin, take_sugar = action
        if take_insulin != 0 and take_sugar != 0: return False
        return True

    def reset(self, *, seed=None, options=None):
        super().reset(seed = seed)
        start_gluco = random.uniform(110, 150) # Partenza leggermente randomizzata
        self.state = np.array([start_gluco, 0, 0, 0, 0, 0, 6, 0, 0, 180], dtype=np.float32)
        self.last_glucose = start_gluco
        
        # Resettiamo solo se serve, per evitare problemi con grafici vuoti se l'episodio finisce
        if len(self.rew_arr) > 100000:
             self.rew_arr = []
             self.gluco_arr = []
             
        return self.state, {}

    def render(self):
        print("")

    def get_res(self):
        return self.rew_arr, self.gluco_arr