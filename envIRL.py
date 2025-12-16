import pandas as pd
import os
import ast
import gymnasium as gym
from gymnasium import spaces
from scipy.special import logsumexp
import numpy as np
import random
from collections import deque

class Gluco_envIRL(gym.Env):

    def __init__(self):
        super(Gluco_envIRL, self).__init__()

        self.action_space = spaces.MultiDiscrete([24,5])  #2 azioni discrete perchè rappresentano scelte finite

        #gli stati invece saranno continui poichè rappresentano grandezze variabili nel tempo
        #Stato: livello glicemia, orario giornata, carboidrati ingeriti, tempo passato dall'ultimo pasto, 
        # insulina presa, tempo passato dall'ultima iniezione, dose insulina basale, attività fisica (valore intero che indichi la tipologia (extra)), 
        # tempo passato dall'attività fisica, valore unità insulinica-glicemia
        #per l'attività fisica valore intero da 0 (nessuna attività) a 3, in base alla complessità e allo sforzo fisico
        self.observation_space = spaces.Box(low = np.array([0,0,0,0,0,0,0,0,0,10], dtype=np.float32),
                                            high = np.array([400,23,300,4,20,4,48,3,23,300], dtype=np.float32),
                                            dtype = np.float32)
        
        self.edges_glucose = np.array([0,30,55,70,90,160,180,220,250,300])
        self.edges_carbs   = np.array([0,30,50,70,90,110])
        self.edges_basal   = np.array([0,6,12,24,48])
        self.edges_sport_t = np.array([0,2,6,24])
        self.edges_IR      = np.array([10,20,50,100,180,300])

        self.state = [100, 0, 0, 0, 0, 0, 6, 0, 0, 90] #stato iniziale, 
        self.rew_arr = []
        self.gluco_arr = []
        self.last_5_glucolevels = deque(maxlen=5) #array che contiene i valori delle ultime 5 glicemie al risveglio (fissare orario intorno alle 7)
        self.day_insulin = 0
        self.learned_rewards = {}
        #self.loaded_trajectories = self.load_expert_trajectories()
        #self.save_expert_trajectories(self.loaded_trajectories)
        self.loaded_trajectories = []
        self.len_episodes = 5000
        self.ranges = [0, 0, 0, 0, 0]
        self.sugar_first = False

    def discretize_state(self, state):
        gluco, hour, carbo, carbo_time, insulin, time_insulin, basal, sport, sport_time, ins_res = state
        
        coords = [
            np.digitize(gluco, self.edges_glucose) - 1,
            int(hour),
            np.digitize(carbo, self.edges_carbs) - 1,
            int(carbo_time),
            int(np.clip(insulin, 0, 20)),
            int(time_insulin),
            np.digitize(basal, self.edges_basal) - 1,
            int(sport),
            np.digitize(sport_time, self.edges_sport_t) - 1,
            np.digitize(ins_res, self.edges_IR) - 1
        ]

        self.dims = [
            len(self.edges_glucose),
            24,
            len(self.edges_carbs),
            5,
            21,
            5,
            len(self.edges_basal),
            4,
            len(self.edges_sport_t),
            len(self.edges_IR)
        ]

        # CLIPPING FONDAMENTALE
        for i, c in enumerate(coords):
            coords[i] = np.clip(c, 0, self.dims[i] - 1)

        return tuple(coords)
        
    def load_expert_trajectories(self, filename="expert_trajectories.csv"):
    
            if not os.path.exists(filename) or os.path.getsize(filename) == 0:
                return []  # Se il file non esiste, restituisce una lista vuota
            
            df = pd.read_csv(filename)

            expert_trajectories = []
            episode = []
            step_count = 0

            for _, row in df.iterrows():
                state = ast.literal_eval(row["state_range"])
                action = list(map(float, row["action"].split(",")))   # Converti l'azione in lista di interi
                episode.append((state, action))
                step_count += 1

                if step_count == 5000:
                    expert_trajectories.append(episode)
                    episode = []
                    step_count = 0


            return expert_trajectories
    
    def filter_invalid_actions(self, action):
        #Elimina combinazioni di azioni non valide
        take_insulin, take_sugar = action
        
        # Non puoi prendere insulina e prendere zuccheri per correggere un'ipoglicemia allo stesso tempo
        if take_insulin != 0 and take_sugar != 0:
            return False

        return True
    
    def step(self, action):

        take_insulin, take_sugar = action

        if not self.filter_invalid_actions(action):
            return self.state, -100, False, False, {}  # Penalità per azioni non valide
        gluco_level, hour, carbo, carbo_time, insulin, time_insulin, basal, sport, sport_time, insulin_resistance = self.state

        if take_insulin != 0:
            insulin = take_insulin*0.5
            self.day_insulin += take_insulin * 0.5
        if take_sugar != 0:
            gluco_level += 36 * take_sugar

        #possibilità di un pasto
        if hour>=7 and hour<=9 and carbo_time==0: #colazione
            if random.random() > 0.4:
                carbo = random.uniform(40,80)
        
        if hour>11 and hour<15 and carbo_time==0: #pranzo
            if random.random() > 0.4:
                carbo = random.uniform(60,100)

        if hour==14 and carbo_time==0:  #pranzo obbligatorio
            carbo = random.uniform(60,100)

        if hour>19 and hour<23 and carbo_time==0: #cena
            if random.random() > 0.4:
                carbo = random.uniform(40,100)

        if hour==22 and carbo_time==0:  #cena obbligatoria
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

        gluco_level = gluco_level + random.uniform(0, 10) #valore randomico di aumento o diminuzione della glicemia a digiuno

        hour = (hour+1)%24

        if hour == 6:     #se l'orario corrisponde alle 7 di mattina registra il valore del glucosio tra quelli più recenti
            self.last_5_glucolevels.append(gluco_level)

            if len(self.last_5_glucolevels) == 5:
                avg_morning = np.mean(self.last_5_glucolevels)

                if avg_morning < 90:
                    basal -= 2  # diminuzione 10%
                elif avg_morning > 140:
                    basal += 2  # aumento 10%

                # Limiti realistici
                basal = np.clip(basal, 5, 40)

                self.last_5_glucolevels.clear()

        if hour == 0:   # a mezzanotte viene calcolato il valore di un'unità di insulina e viene svuotato l'array relativo all'insulina presa durante la giornata
            insulin_resistance = 1800/(self.day_insulin + basal)
            self.day_insulin = 0

        basal_effect = basal * insulin_resistance * 0.01  # effetto insulina basale
        gluco_level -= basal_effect

        gluco_level = np.clip(gluco_level, 0, 400)

        state_tuple = self.discretize_state(np.array([gluco_level, hour, carbo, carbo_time, insulin, time_insulin, basal, sport, sport_time, insulin_resistance], dtype=np.float32))
        reward = self.learned_rewards.get((state_tuple[0]), -10)
        self.rew_arr.append(reward)
        self.gluco_arr.append(gluco_level)
        
        self.state = np.array([gluco_level, hour, carbo, carbo_time, insulin, time_insulin, basal, sport, sport_time, insulin_resistance], dtype=np.float32)   #stato aggiornato
        done = False
        truncated = False

        return self.state, float(reward), done, truncated, {}
    
    def step_IRL(self, action):

        take_insulin, take_sugar = action

        if not self.filter_invalid_actions(action):
            return self.state, -100, False, False, {}  # Penalità per azioni non valide
        
        gluco_level, hour, carbo, carbo_time, insulin, time_insulin, basal, sport, sport_time, insulin_resistance = self.state

        if take_insulin != 0:
            insulin += take_insulin*0.5
            self.day_insulin += take_insulin * 0.5
        if take_sugar != 0:
            gluco_level += 36 * take_sugar
            self.sugar_first = True
        else:
            self.sugar_first = False

        #possibilità di un pasto
        if hour>=7 and hour<=9 and carbo_time==0: #colazione
            if random.random() > 0.4:
                carbo = int(random.uniform(40,80))

        if hour==9 and carbo_time==0:  #colazione obbligatorio
            carbo = int(random.uniform(40,80))
        
        if hour>=11 and hour<14 and carbo_time==0: #pranzo
            if random.random() > 0.4:
                carbo = int(random.uniform(60,100))

        if hour==13 and carbo_time==0:  #pranzo obbligatorio
            carbo = int(random.uniform(60,100))

        if hour>=20 and hour<=22 and carbo_time==0: #cena
            if random.random() > 0.4:
                carbo = int(random.uniform(40,100))

        if hour==22 and carbo_time==0:  #cena obbligatoria
            carbo = int(random.uniform(40,100))

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

        insulin_resistance_t = insulin_resistance * circadian[int(hour)]


        if insulin > 0:
            insulin_profile = [0.2, 0.35, 0.3, 0.15]  # distribuzione più realistica
            if 1 <= time_insulin <= 4:
                idx = int(time_insulin) - 1
                gluco_level -= insulin * insulin_resistance_t * insulin_profile[idx]
            time_insulin += 1
            if time_insulin > 4:
                insulin = 0
                time_insulin = 0

        gluco_level = gluco_level + random.uniform(0, 10) #valore randomico di aumento o diminuzione della glicemia a digiuno

        hour = (hour+1)%24

        if hour == 6:     #se l'orario corrisponde alle 7 di mattina registra il valore del glucosio tra quelli più recenti
            self.last_5_glucolevels.append(gluco_level)

            if len(self.last_5_glucolevels) == 5:
                avg_morning = np.mean(self.last_5_glucolevels)

                if avg_morning < 90:
                    basal -= 2  # diminuzione 10%
                elif avg_morning > 140:
                    basal += 2  # aumento 10%

                # Limiti realistici
                basal = np.clip(basal, 5, 40)

                self.last_5_glucolevels.clear()

        if hour == 0:   # a mezzanotte viene calcolato il valore di un'unità di insulina e viene svuotato l'array relativo all'insulina presa durante la giornata
            insulin_resistance = int(max((1800/(self.day_insulin + basal+1e-6)), 10))
            self.day_insulin = 0

        basal_effect = basal * insulin_resistance_t * 0.005  # effetto insulina basale
        gluco_level -= basal_effect

        gluco_level = int(np.clip(gluco_level, 0, 400))

        self.state = np.array([gluco_level, hour, carbo, carbo_time, insulin, time_insulin, basal, sport, sport_time, insulin_resistance], dtype=np.float32)   #stato aggiornato

        if self.state[0] < 55:
            self.ranges[0] += 1
        if self.state[0] < 70:
            self.ranges[1] += 1
        elif self.state[0] < 180:
            self.ranges[2] += 1
        elif self.state[0] < 220:
            self.ranges[3] += 1
        else:
            self.ranges[4] += 1

        state_tuple = self.discretize_state(np.array([gluco_level, hour, carbo, carbo_time, insulin, time_insulin, basal, sport, sport_time, insulin_resistance], dtype=np.float32))
        reward = self.learned_rewards.get((state_tuple[0]), -10)

        done = False
        truncated = False

        return self.state, float(reward), done, truncated, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed = seed)
        self.state = np.array([100, 0, 0, 0, 0, 0, 6, 0, 0, 90], dtype=np.float32)  #stato iniziale da definire
        return self.state,{}

    def get_res(self):
        return self.rew_arr, self.gluco_arr
    
    def render(self):
        print("")

    def expert_policy(self):        #da migliorare
        gluco_level, hour, carbo, carbo_time, insulin, time_insulin, basal, sport, sport_time, insulin_resistance = self.state
        
        action = [0,0]

        if gluco_level < 30 and not(self.sugar_first):
            action[1] = 3   # 3 porzioni → +108 mg/dL circa
        elif gluco_level < 55 and not(self.sugar_first):
            action[1] = 2   # +72 mg/dL
        elif gluco_level < 70 and not(self.sugar_first):
            if carbo == 0 or carbo_time > 3:
                action[1] = 1   # +36 mg/dL
        elif gluco_level < 90:
            if insulin > 0 and carbo == 0  and not(self.sugar_first):
                action[1] = 1   # +36 mg/dL
            elif carbo > 0 and 1 <= carbo_time <= 3 and (insulin == 0 or time_insulin > 2):
                    action[0] = int(((carbo * 4) / insulin_resistance) * 2 - 1)
        elif gluco_level < 160:
            if gluco_level < 120 and insulin > 0 and carbo == 0  and not(self.sugar_first) and hour!=9 and hour!=13 and hour!=21:
                action[1] = 1   # +36 mg/dL
            if carbo > 0 and 1 <= carbo_time <= 3 and (insulin == 0 or time_insulin > 2):
                action[0] = int(((carbo * 4) / insulin_resistance) * 2) #se dopo due/tre ore dal pasto non è stata iniettata insulina, allora corregge per evitare iperglicemie future
        else:
            if carbo > 0 and 1 <= carbo_time <= 3 and (insulin == 0 or time_insulin > 2):
                action[0] = int(((gluco_level - 110) / insulin_resistance) * 2 + ((carbo * 4) / insulin_resistance) * 2)
            elif (insulin == 0 or time_insulin > 3):
                action[0] = int(((gluco_level - 110) / insulin_resistance) * 2)
        
        
        return (action)

    def save_expert_trajectories(self, expert_trajectories, filename="expert_trajectories.csv"):

        data = []
        
        for episode in expert_trajectories:
            for item in episode:
                if isinstance(item, tuple) and len(item) == 2:
                    state, action = item  # Estrai i valori se il formato è corretto
                else:
                    print(f"Errore: Formato errato per item {item}")  # Debug
                    continue  # Salta elementi malformattati

                state_str = str(state)  # Concatena lo stato come stringa
                action_str = ",".join(map(str, action))  # Concatena l'azione come stringa
                data.append([state_str, action_str])

        # Se ci sono dati validi, salva il file
        if data:
            df = pd.DataFrame(data, columns=["state_range", "action"])

            # Se il file esiste e ha dati, aggiungi senza sovrascrivere
            if os.path.exists(filename) and os.path.getsize(filename) > 0:
                df_existing = pd.read_csv(filename)
                df = pd.concat([df_existing, df], ignore_index=True)

            df.to_csv(filename, index=False)
        else:
            print("Nessuna traiettoria esperta trovata per il salvataggio.")


    def generate_expert_trajectories(self, num_episodes=600):
        expert_trajectories = []
        
        for i in range(num_episodes):
            print(f"\nTraiettoria esperta n.{i}\n")
            self.state, _ = self.reset()       
            self.day_insulin = 0

            episode = []
            for _ in range(self.len_episodes):
                action = self.expert_policy()
                next_state, _, _, _, _ = self.step_IRL(action)
                self.state = next_state
                episode.append((self.discretize_state(self.state), action))
                #episode.append((self.state,action))
            expert_trajectories.append(episode)
        
        return expert_trajectories

    def train_irl(self, num_episodes=600, iterations=500, alpha=0.1):
        expert_trajectories = self.generate_expert_trajectories(num_episodes)
        self.save_expert_trajectories(expert_trajectories)
        #expert_trajectories += self.loaded_trajectories
        print(self.ranges)
        #expert_trajectories = self.loaded_trajectories

        self.maxent_irl(expert_trajectories, iterations, alpha)


    def maxent_irl(self, expert_trajectories, iterations=500, alpha=0.1):   #da fare

        state_indices = {}
        idx = 0

        for traj in expert_trajectories:
            for state, _ in traj:
                s = tuple(state)  # stato già discretizzato!
                if s not in state_indices:
                    state_indices[s] = idx
                    idx += 1

        num_states = len(state_indices)
        state_visits = np.zeros(num_states, dtype=np.float32)
        rewards = np.random.uniform(-0.1, 0.1, num_states)
        policy = np.zeros(num_states, dtype=np.float32)

        # Conta le visite (h, a)
        for traj in expert_trajectories:
            for state, _ in traj:
                state_visits[state_indices[tuple(state)]] += 1

        #state_visits /= np.sum(state_visits) + 1e-6
        state_visits = state_visits / np.max(state_visits + 1e-6)

        for _ in range(iterations):
            policy = np.exp(rewards - logsumexp(rewards))  #distribuzione di probabilità sulle traiettorie
            policy /= (np.sum(policy)+1e-6)

            expected_state_visits = np.zeros_like(state_visits)
            for state_idx in range(num_states):
                if state_idx < len(policy):
                    expected_state_visits[state_idx] = policy[state_idx]
                else:
                    expected_state_visits[state_idx] = policy.mean()

            for state_idx in range(num_states):
                rewards[state_idx] += alpha * (state_visits[state_idx] - expected_state_visits[state_idx])/ (state_visits[state_idx] + 1e-6)

        rewards = np.clip(rewards, -10, None)
        rewards = rewards - np.min(rewards) + 1e-5

        # 1) Raggruppiamo reward per gluco_bin
        gluco_groups = {}  # {gluco_bin: [reward1, reward2, ...]}

        for state, index in state_indices.items():
            gluco_bin = state[0]  # <-- il primo elemento dello stato è l’intervallo di glicemia
            reward = float(rewards[index])

            if gluco_bin not in gluco_groups:
                gluco_groups[gluco_bin] = []
            gluco_groups[gluco_bin].append(reward)

        # 2) Media per ogni intervallo di glicemia
        gluco_rewards = {
            gluco_bin: float(np.mean(r_list))
            for gluco_bin, r_list in gluco_groups.items()
        }

        # 3) Salviamo la versione finale delle reward nel modello
        self.learned_rewards = gluco_rewards

        # 4) Creazione CSV leggibile
        df = pd.DataFrame({
            "gluco_bin": list(gluco_rewards.keys()),
            "reward": list(gluco_rewards.values())
        })

        # Salva in un file CSV
        df.to_csv("learned_rewards.csv", index=False)