import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
import random

class Gluco_env2(gym.Env):

    def __init__(self):
        super(Gluco_env2, self).__init__()

        # --- AZIONI (Mantenute Discrete Multi-Dimensionali) ---
        # 0: Bolus Insulina (0-20 step). 1 step = 0.5 U. Max 10.0 U.
        # 1: Zucchero Rapido (0-3 step). 1 step = 15g.
        self.action_space = spaces.MultiDiscrete([21, 4])

        # --- OSSERVAZIONI ---
        # Per gestire il ritardo fisiologico, l'agente DEVE vedere lo stato nascosto:
        # 0: Glicemia Attuale (Normalizzata)
        # 1: Trend (Variazione rispetto allo step prima)
        # 2: IOB (Insulin On Board) - Quanta insulina è ancora attiva nel corpo
        # 3: COB (Carbs On Board) - Quanti carboidrati devo ancora digerire
        self.observation_space = spaces.Box(
            low=-1.0, high=5.0, shape=(4,), dtype=np.float32
        )

        self.state = [110, 0, 0]
        
        # --- PARAMETRI PAZIENTE VIRTUALE (Realistici) ---
        # ISF (Fattore Sensibilità): 1 U abbassa la glicemia di 40 mg/dL
        self.ISF = 40.0 
        # CR (Rapporto Carboidrati): 15g alzano la glicemia di 35 mg/dL (circa)
        self.CR_factor = 2.3 # mg/dL per grammo
        # Basale necessaria: Il fegato produce ~10 mg/dL/h. Serve 0.25 U/h per coprirlo.
        self.liver_output = 10.0 
        self.ideal_basal = self.liver_output / (self.ISF / 4.0) # Approssimazione per step orario
        
        # Code fisiologiche
        self.active_boluses = [] 
        self.active_meals = []   
        self.active_sugar = []
        
        self.prev_gluco = 120.0
        self.rew_arr = []
        self.gluco_arr = []
        self.hour = 0

    def step(self, action):
        take_insulin_idx, take_sugar_idx = action

        # 1. Decodifica (Valori Reali)
        insulin_dose = take_insulin_idx * 0.5 
        sugar_dose = take_sugar_idx * 15.0     

        # 2. Stacking Azioni (Creazione eventi)
        if insulin_dose > 0:
            self.active_boluses.append({'amount': insulin_dose, 'min_ago': 0})
            self.state[1] = insulin_dose
        
        if sugar_dose > 0:
            # Zucchero rapido (succo) viene assorbito velocemente ma non istantaneamente
            self.active_sugar.append({'amount': sugar_dose, 'min_ago': 0, 'type': 'fast'})
            #self.active_meals.append({'amount': sugar_dose, 'min_ago': 0, 'type': 'fast'})

        # 3. Generazione Pasti (Scenario Realistico)
        # Colazione (8:00), Pranzo (13:00), Cena (20:00) +/- varianza
        
        if self._should_eat_meal():
            # Carbo complessi (assorbimento lento)
            carbs = random.uniform(40, 100)
            self.active_meals.append({'amount': carbs, 'min_ago': 0, 'type': 'slow'})
            self.state[2] = carbs
        '''

        # Colazione (7:00-9:00)
        if 7 <= self.hour <= 9 and len(self.active_meals) == 0:
            if random.random() > 0.4:
                carbo = random.uniform(40, 80)

        if self.hour == 9 and len(self.active_meals) == 0:
            if random.random() > 0.4:
                carbo = random.uniform(40, 80)
        
        # Pranzo (12:00-14:00)
        if 12 <= self.hour <= 14 and len(self.active_meals) == 0:
            if random.random() > 0.4:
                carbo = random.uniform(60, 100)

        # Pranzo Obbligatorio (14:00)
        if self.hour == 14 and len(self.active_meals) == 0:
            carbo = random.uniform(60, 100)

        # Cena (20:00-22:00)
        if 20 <= self.hour <= 22 and len(self.active_meals) == 0:
            if random.random() > 0.4:
                carbo = random.uniform(40, 100)

        # Cena Obbligatoria (22:00)
        if self.hour == 22 and len(self.active_meals) == 0:
            carbo = random.uniform(40, 100)

        carbo = random.uniform(40, 90)
        self.active_meals.append({'amount': carbo, 'min_ago': 0, 'type': 'slow'})
        self.state[2] = carbo
        '''
        # 4. SIMULAZIONE FISIOLOGICA (Il cuore del realismo)
        # Simuliamo step di 1 ora (o frazioni se necessario, qui semplifichiamo a step discreti)
        # Ma usiamo curve di farmacocinetica realistiche.

        # --- A. Produzione Epatica vs Basale ---
        # Il fegato alza la glicemia costantemente. La basale (qui assunta perfetta o gestita altrove) la abbassa.
        # Assumiamo che l'agente controlli solo i BOLI. La basale è fissa e copre il fegato.
        # Introduciamo un leggero "drift" (errore basale) casuale.
        basal_drift = random.uniform(-10, 10) #cambio della glicemia in un'ora
        self.gluco_level += basal_drift 

        # --- B. Curva Insulina (Novorapid/Humalog Model) ---
        # Durata azione: ~4-5 ore. Picco: 60-90 min.
        # Profilo discretizzato su 5 ore: [5%, 25%, 40%, 20%, 10%]
        # NOTA: Questo ritardo rende il training difficile ma realistico.
        insulin_curve = [0.05, 0.25, 0.40, 0.20, 0.10]
        
        total_iob = 0.0
        active_boluses_next = []
        
        for bolus in self.active_boluses:
            t = bolus['min_ago'] # in ore
            dose = bolus['amount']
            
            # Calcolo IOB (quanto ne resta da assorbire in futuro)
            remaining_pct = sum(insulin_curve[t+1:]) if t+1 < len(insulin_curve) else 0
            total_iob += dose * remaining_pct
            
            # Effetto attuale
            if t < len(insulin_curve):
                # Quanto scende la glicemia: Dose * ISF * %curva
                drop = dose * self.ISF * insulin_curve[t]
                self.gluco_level -= drop
            
            bolus['min_ago'] += 1
            if bolus['min_ago'] <= len(insulin_curve):
                active_boluses_next.append(bolus)
        
        self.active_boluses = active_boluses_next

        if len(self.active_boluses) == 0:
            self.state[1] = 0

        # --- C. Curva Carboidrati ---
        # Fast (Zucchero): [60%, 40%] - 2 ore
        # Slow (Pasto): [10%, 30%, 40%, 20%] - 4 ore (Digestione lenta)
        fast_curve = [0.6, 0.4]
        slow_curve = [0.1, 0.3, 0.4, 0.2]

        total_cob = 0.0
        active_meals_next = []
        active_sugar_next = []

        for meal in self.active_meals:
            t = meal['min_ago']
            grams = meal['amount']
            curve = fast_curve if meal['type'] == 'fast' else slow_curve
            
            remaining_pct = sum(curve[t+1:]) if t+1 < len(curve) else 0
            total_cob += grams * remaining_pct
            
            if t < len(curve):
                # Quanto sale la glicemia
                rise = grams * self.CR_factor * curve[t]
                self.gluco_level += rise
            
            meal['min_ago'] += 1
            if meal['min_ago'] <= len(curve):
                active_meals_next.append(meal)

        for sugar in self.active_sugar:
            t = sugar['min_ago']
            grams = sugar['amount']
            curve = fast_curve
            remaining_pct = sum(curve[t+1:]) if t+1 < len(curve) else 0
            total_cob += grams * remaining_pct
            
            if t < len(curve):
                # Quanto sale la glicemia
                rise = grams * self.CR_factor * curve[t]
                self.gluco_level += rise
            
            sugar['min_ago'] += 1
            if sugar['min_ago'] <= len(curve):
                active_sugar_next.append(sugar)
        
        self.active_meals = active_meals_next
        self.active_sugar = active_sugar_next

        if len(self.active_meals) == 0:
            self.state[2] = 0

        # Clipping e Trend
        self.gluco_level = np.clip(self.gluco_level, 0, 600)
        self.state[0] = self.gluco_level
        trend = self.gluco_level - self.prev_gluco
        self.prev_gluco = self.gluco_level
        self.hour = (self.hour + 1) % 24

        # 5. REWARD FUNCTION (Adattata al ritardo)
        terminated = False
        truncated = False
        
        # Morte Clinica
        if self.gluco_level < 20 or self.gluco_level >= 590:
            terminated = True
            reward = -100.0
        else:
            reward = self._calculate_reward(self.gluco_level, trend, total_iob)
            # Costo azione (piccolo)
            if insulin_dose > 0: reward -= 0.1

        self.rew_arr.append(reward)
        self.gluco_arr.append(self.gluco_level)
        
        return self._get_obs(total_iob, total_cob, trend), float(reward), terminated, truncated, self.state #per il test inserire self.state

    def _should_eat_meal(self):
        # Generatore pasti probabilistico basato sull'ora
        h = self.hour
        prob = 0.0
        if 7 <= h < 9: prob = 0.2 # Colazione
        elif h == 9: prob = 1
        elif 12 <= h < 14: prob = 0.2 # Pranzo
        elif h == 14: prob = 1
        elif 19 <= h < 21: prob = 0.2 # Cena
        elif h == 21: prob = 1
        
        # Evita di mangiare se c'è già un pasto attivo (per non sovrapporre troppo nel training)
        if len(self.active_meals) > 0:
            prob = 0.0
            
        return random.random() < prob

    def _calculate_reward(self, bg, trend, iob):
        # Obiettivo Clinico: Time in Range (70-180), ottimale 110.
        
        # 1. Distanza dal target (Gaussiana)
        error = abs(bg - 110)
        r = np.exp(- (error / 50.0)**2) * 2.0 # Max +2
        
        # 2. Penalità Ipo/Iper
        if bg < 70: 
            r -= (70 - bg) * 0.2 # Ipo penalizzata pesantemente
        if bg > 180:
            r -= (bg - 180) * 0.05 # Iper penalizzata linearmente
            
        # 3. Penalità "Safety" su IOB (CRUCIALE per realismo)
        # Se la glicemia sta scendendo o è bassa, E ho ancora molta insulina attiva,
        # significa che andrò in ipoglicemia tra 1 ora. DEVO penalizzare ORA.
        predicted_bg = bg + trend - (iob * self.ISF * 0.5) # Stima rozza futura
        if predicted_bg < 50:
            r -= 2.0 # Penalità predittiva
            
        return r

    def _get_obs(self, iob, cob, trend):
        # Normalizzazione: Centrato su 110, diviso per 200 (range fisiologico)
        obs_gluco = (self.gluco_level - 110.0) / 200.0
        obs_trend = trend / 50.0
        obs_iob = iob / 10.0 # Normalizzato su 10U
        obs_cob = cob / 100.0 # Normalizzato su 100g
        
        return np.array([obs_gluco, obs_trend, obs_iob, obs_cob], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.hour = random.randint(0, 23)
        self.gluco_level = random.uniform(100, 160)
        self.prev_gluco = self.gluco_level
        self.active_boluses = []
        self.active_meals = []
        self.active_sugar = []
        #self.rew_arr = []
        #self.gluco_arr = []
        return self._get_obs(0,0,0), {}

    def render(self):
        pass

    def get_res(self):
        return self.rew_arr, self.gluco_arr