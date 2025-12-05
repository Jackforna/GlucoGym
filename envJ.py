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
        # tempo passato dall'attività fisica, valore unità insulinica-glicemia, unità insulina per pasto
        #per l'attività fisica valore intero da 0 (nessuna attività) a 3, in base alla complessità e allo sforzo fisico
        self.observation_space = spaces.Box(low = np.array([0,0,0,0,0,0,0,0,0,15,0], dtype=np.float32),
                                            high = np.array([400,23,3000,4,25,4,48,3,23,200,25], dtype=np.float32),
                                            dtype = np.float32)

        self.state = [100, 0, 0, 0, 0, 0, 6, 0, 0, 180, 2] #stato iniziale, 
        self.rew_arr = []
        self.gluco_arr = []
        self.last_5_glucolevels = [] #array che contiene i valori delle ultime 5 glicemie al risveglio (fissare orario intorno alle 7)
        self.day_insulin = []
    
    def step(self, action):

        take_insulin, take_sugar = action
        if not self.filter_invalid_actions(action):
            return self.state, -100, False, False, {}  # Penalità per azioni non valide
        gluco_level, hour, carbo, carbo_time, insulin, time_insulin, basal, sport, sport_time, insulin_resistance, insulin_for_meal = self.state

        if take_insulin != 0:
            time_insulin = 1
            insulin = take_insulin*0.5
            self.day_insulin.append(insulin)

            '''if carbo != 0:
                self.day_insulin.append(insulin_for_meal)   #insulina da pasto
            else:
                unit = round((100 - gluco_level)/max(insulin_resistance, 1e-6) * 2)/2 #deve avere valore intero o da mezza unità
                self.day_insulin.append(unit)    '''           #insulina correttiva
        elif take_sugar != 0:
            gluco_level += random.uniform(30,60) * take_sugar
        else:
            ...
        
        reward = np.exp(-0.5 * ((gluco_level - 110) / 30) ** 2) #funzione gaussiana per il calcolo della reward

        if gluco_level<70:  #aggiustamenti per valori esterni all'intervallo ideale
            reward -= 1
        elif gluco_level >250:
            reward -= 0.5

        #possibilità di un pasto
        if hour>=7 and hour<=9 and carbo_time==0: #colazione
            x = random.uniform(0,1)
            if x==1:
                carbo_time = 1
                carbo = random.uniform(40,80)
        
        if hour>11 and hour<15 and carbo_time==0: #pranzo
            x = random.uniform(0,1)
            if x==1:
                carbo_time = 1
                carbo = random.uniform(60,100)

        if hour==14 and carbo_time==0:  #pranzo obbligatorio
            carbo_time = 1
            carbo = random.uniform(60,100)

        if hour>19 and hour<23 and carbo_time==0: #cena
            x = random.uniform(0,1)
            if x==1:
                carbo_time = 1
                carbo = random.uniform(40,100)

        if hour==22 and carbo_time==0:  #cena obbligatoria
            carbo_time = 1
            carbo = random.uniform(40,100)

        if insulin > 1.5:
            insulin_for_meal = insulin

        if carbo!=0 and carbo_time==2:      #aumento dei livelli di glucosio un'ora dal pasto(aggiungere aumento graduale nelle ore successive)
            gluco_level += (carbo-70)/10*50 + insulin_for_meal*insulin_resistance

            
        carbo_time += 1

        if time_insulin == 1:   #dopo due ore inizia l'effetto dell'insulina(aggiungere diminuzione graduale nelle ore successive)
            gluco_level -= insulin * insulin_resistance/2
            #gluco_level += random.uniform(-20, 40)
        elif time_insulin == 2:
            gluco_level -= insulin * insulin_resistance/2
        elif time_insulin == 4:
            time_insulin = 0
            insulin = 0

        #gluco_level = np.clip((gluco_level + random.uniform(-20, 20)), 0 , 400) #valore randomico di aumento o diminuzione della glicemia a digiuno

        self.rew_arr.append(reward)
        self.gluco_arr.append(gluco_level)

        if hour == 23:
            hour = 0
        else:
            hour += 1

        if carbo_time == 4:
            carbo = 0
            carbo_time = 0

        self.last_5_glucolevels = deque(maxlen=5) #tiene solamente gli ultimi 5 valori nell'array

        if hour == 6:     #se l'orario corrisponde alle 7 di mattina registra il valore del glucosio tra quelli più recenti
            self.last_5_glucolevels.append(gluco_level)

        if hour == 0:   # a mezzanotte viene calcolato il valore di un'unità di insulina e viene svuotato l'array relativo all'insulina presa durante la giornata
            insulin_resistance = 1800/(sum(self.day_insulin) + basal)
            self.day_insulin.clear()
        
        self.state = np.array([gluco_level, hour, carbo, carbo_time, insulin, time_insulin, basal, sport, sport_time, insulin_resistance, insulin_for_meal], dtype=np.float32)   #stato aggiornato
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
        self.state = np.array([100, 0, 0, 0, 0, 0, 6, 0, 0, 180, 1.5], dtype=np.float32)  #stato iniziale da definire
        return self.state,{}


    def render(self):
        print("")

    def get_res(self):
        return self.rew_arr, self.gluco_arr