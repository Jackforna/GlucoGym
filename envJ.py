import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
import random

class Gluco_env(gym.Env):

    def __init__(self):
        super(Gluco_env, self).__init__()

        self.action_space = spaces.MultiDiscrete([24,5])  #2 azioni discrete perchè rappresentano scelte finite

        #gli stati invece saranno continui poichè rappresentano grandezze variabili nel tempo
        #Stato: livello glicemia, orario giornata, carboidrati ingeriti, tempo passato dall'ultimo pasto, 
        # insulina presa, tempo passato dall'ultima iniezione, dose insulina basale, attività fisica (valore intero che indichi la tipologia (extra)), 
        # tempo passato dall'attività fisica, valore unità insulinica-glicemia
        #per l'attività fisica valore intero da 0 (nessuna attività) a 3, in base alla complessità e allo sforzo fisico
        self.observation_space = spaces.Box(low = np.array([0,0,0,0,0,0,0,0,0,10], dtype=np.float32),
                                            high = np.array([400,23,300,4,20,4,48,3,23,200], dtype=np.float32),
                                            dtype = np.float32)

        self.state = [100, 0, 0, 0, 0, 0, 6, 0, 0, 50] #stato iniziale, 
        self.rew_arr = []
        self.gluco_arr = []
        self.last_5_glucolevels = deque(maxlen=5) #array che contiene i valori delle ultime 5 glicemie al risveglio (fissare orario intorno alle 7)
        self.day_insulin = []
    
    def step(self, action):

        take_insulin, take_sugar = action

        if not self.filter_invalid_actions(action):
            return self.state, -100, False, False, {}  # Penalità per azioni non valide
        gluco_level, hour, carbo, carbo_time, insulin, time_insulin, basal, sport, sport_time, insulin_resistance = self.state

        if take_insulin != 0:
            time_insulin = 1
            insulin = take_insulin*0.5
            self.day_insulin.append(insulin)
        if take_sugar != 0:
            gluco_level += 36 * take_sugar

        #possibilità di un pasto
        if hour>=7 and hour<=9 and carbo_time==0: #colazione
            x = random.random()
            if x>0.4:
                carbo_time = 1
                carbo = random.uniform(40,80)
        
        if hour>11 and hour<15 and carbo_time==0: #pranzo
            x = random.random()
            if x>0.4:
                carbo_time = 1
                carbo = random.uniform(60,100)

        if hour==14 and carbo_time==0:  #pranzo obbligatorio
            carbo_time = 1
            carbo = random.uniform(60,100)

        if hour>19 and hour<23 and carbo_time==0: #cena
            x = random.random()
            if x>0.4:
                carbo_time = 1
                carbo = random.uniform(40,100)

        if hour==22 and carbo_time==0:  #cena obbligatoria
            carbo_time = 1
            carbo = random.uniform(40,100)

        #aumento glicemia in base ai pasti

        if carbo > 0: #da migliorare
            if carbo_time == 1: #4 mg/dL per grammo di carboidrati
                gluco_level += 0.1 * carbo * 4
            elif carbo_time == 2:
                gluco_level += 0.35 * carbo * 4
            elif carbo_time == 3:
                gluco_level += 0.35 * carbo * 4
            elif carbo_time == 4:
                gluco_level += 0.2 * carbo * 4

            carbo_time += 1

            if carbo_time > 4:
                carbo = 0
                carbo_time = 0

        # variazione circadiana, la resistenza insulinica cambia in base all'orario della giornata
        circadian = [1.2, 1.1, 1.0, 0.9, 0.85, 0.9, 1.0, 1.1, 1.2, 1.25,
                    1.2, 1.1, 1.0, 0.95, 1.0, 1.1, 1.15, 1.2, 1.15, 1.1,
                    1.0, 0.95, 0.9, 1.0]

        insulin_resistance = insulin_resistance * circadian[int(hour)]


        if time_insulin > 0:
            insulin_profile = [0.2, 0.35, 0.30, 0.15]  # distribuzione più realistica
            if 1 <= time_insulin <= 4:
                idx = int(time_insulin) - 1
                gluco_level -= insulin * insulin_resistance * insulin_profile[idx]
            time_insulin += 1
            if time_insulin > 4:
                insulin = 0
                time_insulin = 0

        gluco_level = gluco_level + random.uniform(-10, 10) #valore randomico di aumento o diminuzione della glicemia a digiuno

        hour = (hour+1)%24

        if hour == 6:     #se l'orario corrisponde alle 7 di mattina registra il valore del glucosio tra quelli più recenti
            self.last_5_glucolevels.append(gluco_level)

            if len(self.last_5_glucolevels) == 5:
                avg_morning = np.mean(self.last_5_glucolevels)

                if avg_morning < 90:
                    basal *= 0.9  # diminuzione 10%
                elif avg_morning > 140:
                    basal *= 1.1  # aumento 10%

                # Limiti realistici
                basal = np.clip(basal, 5, 40)

                self.last_5_glucolevels.clear()

        if hour == 0:   # a mezzanotte viene calcolato il valore di un'unità di insulina e viene svuotato l'array relativo all'insulina presa durante la giornata
            insulin_resistance = 1800/(sum(self.day_insulin) + basal)
            self.day_insulin.clear()

        basal_effect = basal * insulin_resistance * 0.01  # effetto insulina basale
        gluco_level -= basal_effect

        gluco_level = np.clip(gluco_level, 0, 400)

        
        reward = np.exp(-0.5 * ((gluco_level - 110) / 18) ** 2) #funzione gaussiana per il calcolo della reward

        '''if gluco_level < 70:
            reward -= 5
        if gluco_level < 55:
            reward -= 15
        if gluco_level < 30:
            reward -= 100
        if gluco_level > 250:
            reward -= 15
        if gluco_level > 300:
            reward -= 50
        if gluco_level > 85 and gluco_level < 160:
            reward += 10

        if gluco_level > 100 and take_sugar>0:
            reward -= 5

        if gluco_level > 250 and take_insulin<1:
            reward -= 2'''
        
        # ipoglicemia: penalità più morbida (evita panico)
        reward -= max(0, (70 - gluco_level) / 20)

        # iperglicemia
        reward -= max(0, (gluco_level - 180) / 50)

        # ultra-reward nel range perfetto 90–140
        if 90 <= gluco_level <= 140:
            reward += 3
        
        self.rew_arr.append(reward)
        self.gluco_arr.append(gluco_level)
        
        self.state = np.array([gluco_level, hour, carbo, carbo_time, insulin, time_insulin, basal, sport, sport_time, insulin_resistance], dtype=np.float32)   #stato aggiornato
        done = False
        truncated = False
        

        return self.state, float(reward), done, truncated, {}
    
    def filter_invalid_actions(self, action):
        #Elimina combinazioni di azioni non valide
        take_insulin, take_sugar = action
        
        # Non puoi prendere insulina e prendere zuccheri per correggere un'ipoglicemia allo stesso tempo
        if take_insulin != 0 and take_sugar != 0:
            return False

        return True

    def reset(self, *, seed=None, options=None):
        super().reset(seed = seed)
        self.state = np.array([100, 0, 0, 0, 0, 0, 6, 0, 0, 180], dtype=np.float32)  #stato iniziale da definire
        return self.state,{}


    def render(self):
        print("")

    def get_res(self):
        return self.rew_arr, self.gluco_arr