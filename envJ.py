import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

class Gluco_env(gym.Env):

    def __init__(self):
        super(Gluco_env, self).__init__()

        self.action_space = spaces.MultiDiscrete([2,2])  #2 azioni discrete perchè rappresentano scelte finite

        #gli stati invece saranno continui poichè rappresentano grandezze variabili nel tempo
        #Stato: livello glicemia, orario giornata, carboidrati ingeriti, tempo passato dall'ultimo pasto, 
        # insulina presa, tempo passato dall'ultima iniezione, dose insulina basale, attività fisica (valore intero che indichi la tipologia (extra)), 
        # tempo passato dall'attività fisica
        #per l'attività fisica valore intero da 0 (nessuna attività) a 3, in base alla complessità e allo sforzo fisico
        self.observation_space = spaces.Box(low = np.array([0,0,0,0,0,0,0,0,0], dtype=np.float32),
                                            high = np.array([400,23,3000,4,25,4,48,3,23], dtype=np.float32),
                                            dtype = np.float32)

        self.state = [100, 0, 0, 0, 0, 0, 6, 0, 0] #stato iniziale, 
        self.rew_arr = []
        self.gluco_arr = []
        self.last_5_glucolevels = [] #array che contiene i valori delle ultime 5 glicemie al risveglio (fissare orario intorno alle 7)
    
    def step(self, action):

        take_insulin, take_carbo = action
        if not self.filter_invalid_actions(action):
            return self.state, -100, False, False, {}  # Penalità per azioni non valide
        gluco_level, hour, carbo, carbo_time, insulin, time_insulin, basal, sport, sport_time = self.state

        if take_insulin == 1:
            gluco_level = ...
        elif take_carbo == 1:
            gluco_level = ...
        else:
            ...
        
        if gluco_level < 50:
            reward = ...
        elif gluco_level < 70:
            reward = ...
        elif gluco_level < 180:
            reward = ...
        elif gluco_level < 220:
            reward = ...
        elif gluco_level < 250:
            reward = ...
        else:
            reward = ...


        self.rew_arr.append(reward)
        self.gluco_arr.append(gluco_level)

        if hour == 23:
            hour = 0
        else:
            hour += 1

        if take_insulin == 1:
            time_insulin = 0
        else:
            time_insulin += 1

        if take_carbo == 1:
            carbo_time = 0
        else:
            carbo_time += 1

        self.last_5_glucolevels = deque(maxlen=5) #tiene solamente gli ultimi 5 valori nell'array

        if hour == 6:     #se l'orario corrisponde alle 7 di mattina registra il valore del glucosio tra quelli più recenti
            self.last_5_glucolevels.append(gluco_level)
        
        self.state = np.array([gluco_level, hour, carbo, carbo_time, insulin, time_insulin, basal, sport, sport_time])   #stato aggiornato
        done = False
        truncated = False
        

        return self.state, float(reward), done, truncated, {}
    
    def filter_invalid_actions(self, action):
        #Elimina combinazioni di azioni non valide
        take_insulin, take_carbo = action
        
        # Non puoi prendere insulina e prendere zuccheri per correggere un'ipoglicemia allo stesso tempo
        if take_insulin == 1 and take_carbo == 1:
            return False

        return True

    def reset(self, *, seed=None, options=None):
        super().reset(seed = seed)
        self.state = np.array([100, 0, 0, 0, 0, 0, 6, 0, 0], dtype=np.float32)  #stato iniziale da definire
        return self.state,{}


    def render(self):
        print("")

    def get_res(self):
        return self.rew_arr, self.gluco_arr