import gymnasium as gym
from gymnasium import spaces
import numpy as np

class Gluco_env(gym.Env):

    def __init__(self):
        super(Gluco_env, self).__init__()

        self.action_space = spaces.MultiDiscrete([2,2])  #2 azioni discrete perchè rappresentano scelte finite

        #gli stati invece saranno continui poichè rappresentano grandezze variabili nel tempo
        #Stato: ...
        self.observation_space = spaces.Box(low = np.array([...], dtype=np.float32),
                                            high = np.array([...], dtype=np.float32),
                                            dtype = np.float32)

        self.state = [...] #stato iniziale
        self.rew_arr = []
        self.gluco_arr = []
    
    def step(self, action):

        take_insulin, take_carbo = action
        if not self.filter_invalid_actions(action):
            return self.state, -100, False, False, {}  # Penalità per azioni non valide
        ... = self.state

        gluco_level = ... #da calcolare in base all'azione
        
        reward = ...

        self.rew_arr.append(reward)
        self.gluco_arr.append(gluco_level)
        
        self.state = ...   #stato aggiornato
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
        self.state = np.array([...], dtype=np.float32)  #stato iniziale da definire
        return self.state,{}


    def render(self):
        print("")

    def get_res(self):
        return self.rew_arr, self.gluco_arr