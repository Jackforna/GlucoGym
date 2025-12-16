import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
import random

class Gluco_env(gym.Env):

    def __init__(self):
        super(Gluco_env, self).__init__()
        
        # --- Costanti di Scala ---
        self.MINUTES_PER_STEP = 5 
        self.STEPS_PER_HOUR = 60 / self.MINUTES_PER_STEP  # 12 steps/ora
        self.STEPS_PER_DAY = 24 * self.STEPS_PER_HOUR    # 288 steps/giorno

        # La durata massima in ore (24 ore * 5 minuti/step) per l'azione dell'insulina/carbo/sport
        # Deve essere convertita in numero di steps (es. 4 ore = 4 * 12 = 48 steps)
        
        # Action Space (Mantenuto, ma l'impatto è ricalibrato in step)
        self.action_space = spaces.MultiDiscrete([24, 5]) 

        # Observation Space (Il tempo viene scalato in steps 0-287)
        # Stato: [gluco, step_day, carbo, carbo_steps, insulin, time_insulin_steps, basal, sport, sport_steps, IR]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 10], dtype=np.float32),
            high=np.array([400, self.STEPS_PER_DAY - 1, 300, 4 * self.STEPS_PER_HOUR, 20, 4 * self.STEPS_PER_HOUR, 48, 3, 23 * self.STEPS_PER_HOUR, 300], dtype=np.float32),
            dtype=np.float32
        )

        self.state = np.array([100, 0, 0, 0, 0, 0, 6, 0, 0, 180], dtype=np.float32)
        self.rew_arr = []
        self.gluco_arr = []
        self.last_5_glucolevels = deque(maxlen=5)
        self.day_insulin = []
        self.current_step_of_day = 0  # 0 to 287

    
    def step(self, action):
        
        take_insulin, take_sugar = action

        steps_per_phase = int(self.STEPS_PER_HOUR) # 12

        if not self.filter_invalid_actions(action):
            return self.state, -100.0, False, False, {}

        # Scompatta lo stato (Nota: 'hour' è ora 'step_day')
        gluco_level, step_day, carbo, carbo_time_steps, insulin, time_insulin_steps, basal, sport, sport_time_steps, insulin_resistance = self.state

        # --- Aggiornamento Azioni (Insulina/Zucchero) ---
        if take_insulin != 0:
            # Durata dell'insulina: 4 ore = 4 * 12 = 48 steps
            time_insulin_steps = 1
            insulin = take_insulin * 0.5
            self.day_insulin.append(insulin)
        if take_sugar != 0:
            gluco_level += 36 * take_sugar

        # --- Logica Pasti (Convertita in Steps) ---
        
        # Le ore sono convertite in range di step. Esempio: 7:00-9:00 = step 84-108
        hour_of_day = step_day / self.STEPS_PER_HOUR

        # Colazione (7:00-9:00)
        if 7 <= hour_of_day <= 9 and carbo_time_steps == 0:
            if random.random() > 0.4:
                carbo_time_steps = 1
                carbo = random.uniform(40, 80)
        
        # Pranzo (12:00-14:00)
        if 12 <= hour_of_day <= 14 and carbo_time_steps == 0:
            if random.random() > 0.4:
                carbo_time_steps = 1
                carbo = random.uniform(60, 100)

        # Pranzo Obbligatorio (14:00)
        if 14 <= hour_of_day < 14 + (1/self.STEPS_PER_HOUR) and carbo_time_steps == 0:
            carbo_time_steps = 1
            carbo = random.uniform(60, 100)

        # Cena (20:00-22:00)
        if 20 <= hour_of_day <= 22 and carbo_time_steps == 0:
            if random.random() > 0.4:
                carbo_time_steps = 1
                carbo = random.uniform(40, 100)

        # Cena Obbligatoria (22:00)
        if 22 <= hour_of_day < 22 + (1/self.STEPS_PER_HOUR) and carbo_time_steps == 0:
            carbo_time_steps = 1
            carbo = random.uniform(40, 100)

        # --- Aumento Glicemia (Nuova Scala) ---

        # Durata assimilazione carbo: 4 ore (48 steps)
        CARBO_DURATION_STEPS = 4 * self.STEPS_PER_HOUR # 48 steps totali
        
        if carbo > 0 and carbo_time_steps <= CARBO_DURATION_STEPS:
            # Nuovi profili di assorbimento su 48 steps
            # Total absorption = 0.1+0.35+0.35+0.2 = 1.0 (corretto)
            
            # Quanti steps per ogni fase di 1 ora (12 steps)
            steps_per_phase = int(self.STEPS_PER_HOUR) 
            
            phase_profile = [0.1, 0.35, 0.35, 0.2] # % del totale assorbita per ogni ora
            phase = int(carbo_time_steps // steps_per_phase)
            
            if phase < len(phase_profile):
                # La variazione deve essere divisa per il numero di step nella fase
                absorption_rate_per_step = (phase_profile[phase] * 4) / steps_per_phase 
                gluco_level += absorption_rate_per_step * carbo
                
            carbo_time_steps += 1

            if carbo_time_steps > CARBO_DURATION_STEPS:
                carbo = 0
                carbo_time_steps = 0

        # --- Variazione Circadiana (Nuova Scala) ---
        
        # Manteniamo l'array orario e usiamo l'interpolazione o il bin più vicino
        circadian = [1.2, 1.1, 1.0, 0.9, 0.85, 0.9, 1.0, 1.1, 1.2, 1.25,
                     1.2, 1.1, 1.0, 0.95, 1.0, 1.1, 1.15, 1.2, 1.15, 1.1,
                     1.0, 0.95, 0.9, 1.0]
        
        # Usa l'indice orario più vicino per l'effetto circadiano
        circadian_idx = int(round(hour_of_day)) % 24 
        
        # La resistenza cambia MOLTO meno ad ogni step
        insulin_resistance = insulin_resistance * (1 + (circadian[circadian_idx] - 1) / self.STEPS_PER_HOUR)

        # --- Effetto Insulina (Nuova Scala) ---
        
        # Durata insulina: 4 ore = 48 steps
        INSULIN_DURATION_STEPS = 4 * self.STEPS_PER_HOUR # 48 steps totali
        
        if time_insulin_steps > 0 and time_insulin_steps <= INSULIN_DURATION_STEPS:
            
            # Profilo dell'insulina su 48 steps: distribuzione cumulativa per ogni ora (4 * 12 steps)
            insulin_profile = [0.2, 0.35, 0.30, 0.15] # % del totale per ogni ora (12 steps)
            
            # Qual è la fase oraria corrente (0, 1, 2, o 3)
            phase = int(time_insulin_steps // steps_per_phase)
            
            if phase < len(insulin_profile):
                # L'effetto deve essere distribuito sugli steps di quella fase (12 steps)
                effect_per_step = (insulin_profile[phase] / steps_per_phase)
                
                # Formula di correzione: Effetto ridotto per step
                gluco_level -= insulin * insulin_resistance * effect_per_step
            
            time_insulin_steps += 1
            
            if time_insulin_steps > INSULIN_DURATION_STEPS:
                insulin = 0
                time_insulin_steps = 0

        # Variazione randomica (molto più piccola su 5 minuti)
        gluco_level = gluco_level + random.uniform(-10/self.STEPS_PER_HOUR, 10/self.STEPS_PER_HOUR) 

        # --- Aggiornamento del Tempo e Logiche Giornaliere ---
        step_day = (step_day + 1) % self.STEPS_PER_DAY

        # Ora di risveglio (6:00, step 72)
        if step_day == 6 * self.STEPS_PER_HOUR: 
            self.last_5_glucolevels.append(gluco_level)

            if len(self.last_5_glucolevels) == 5:
                avg_morning = np.mean(self.last_5_glucolevels)

                # Aggiornamento Basale (Logica giornaliera)
                if avg_morning < 90:
                    basal -= 2 
                elif avg_morning > 140:
                    basal += 2 

                basal = np.clip(basal, 5, 40)
                self.last_5_glucolevels.clear()

        # Aggiornamento IR (Mezzanotte, step 0)
        if step_day == 0:  
            epsilon = 1e-6
            denominatore = sum(self.day_insulin) + basal + epsilon
            # La formula del TDI (1800/TDI) è corretta per il TDI totale del giorno precedente
            insulin_resistance = max(1800 / denominatore, 10) 
            self.day_insulin.clear()

        # Effetto Basale (Basal) - Distribuito per step
        basal_effect = basal * insulin_resistance * 0.01 / self.STEPS_PER_HOUR # Diviso per 12
        gluco_level -= basal_effect

        gluco_level = np.clip(gluco_level, 0, 400)

        # --- Calcolo Reward (Invariato) ---
        reward = np.exp(-0.5 * ((gluco_level - 110) / 18) ** 2) 
        reward -= max(0, (70 - gluco_level) / 20)
        reward -= max(0, (gluco_level - 180) / 50)
        if gluco_level < 110 and take_insulin > 0:
            reward -= 15
        if 90 <= gluco_level <= 160:
            reward += 10
        elif 70 < gluco_level < 90 or 160 < gluco_level < 179:
            reward += 5
        elif 180 < gluco_level < 220:
            reward += 1
        elif gluco_level < 70 or gluco_level > 220:
            reward -= 2
        elif gluco_level < 55 or gluco_level > 300:
            reward -= 75

        self.rew_arr.append(reward)
        self.gluco_arr.append(gluco_level)

        # --- Aggiornamento Stato e Ritorno ---
        self.state = np.array([gluco_level, step_day, carbo, carbo_time_steps, insulin, time_insulin_steps, basal, sport, sport_time_steps, insulin_resistance], dtype=np.float32) 
        
        done = False
        truncated = False

        return self.state, float(reward), done, truncated, {}


    def filter_invalid_actions(self, action):
        take_insulin, take_sugar = action
        if take_insulin != 0 and take_sugar != 0:
            return False
        return True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Stato iniziale (gluco, step_day, carbo, carbo_steps, insulin, time_insulin_steps, basal, sport, sport_steps, IR)
        self.state = np.array([100, 0, 0, 0, 0, 0, 6, 0, 0, 180], dtype=np.float32)
        self.current_step_of_day = 0
        self.day_insulin.clear()
        self.last_5_glucolevels.clear()
        
        return self.state, {}
    
    def render(self):

        print("")



    def get_res(self):

        return self.rew_arr, self.gluco_arr