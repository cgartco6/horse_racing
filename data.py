# data_simulator.py
import numpy as np
import pandas as pd
import random
from datetime import datetime

def simulate_race_data(num_races=5, num_horses_per_race=8):
    races = []
    for race_id in range(1, num_races+1):
        race_date = datetime.now().date()
        race_time = f"{random.randint(13,19)}:{random.randint(0,59):02d}"
        track = random.choice(['Ascot', 'Epsom', 'Goodwood', 'York', 'Newmarket'])
        weather = random.choice(['Good', 'Soft', 'Heavy', 'Firm'])
        distance = random.choice([1000, 1200, 1400, 1600, 2000, 2400])
        
        horses = []
        for horse_id in range(1, num_horses_per_race+1):
            horse_name = f"Horse_{race_id}_{horse_id}"
            horse_age = random.randint(3, 8)
            jockey_win_rate = random.uniform(0.1, 0.5)
            trainer_win_rate = random.uniform(0.1, 0.5)
            weight = random.randint(50, 65)
            # Simulate some features that might affect performance
            recent_form = random.uniform(0, 1)  # 0 to 1 scale, higher is better
            days_since_last_run = random.randint(10, 100)
            # We'll simulate an injury as a binary (0 or 1) but with low probability
            injury = 1 if random.random() < 0.1 else 0
            
            horse_data = {
                'race_id': race_id,
                'horse_name': horse_name,
                'horse_age': horse_age,
                'jockey_win_rate': jockey_win_rate,
                'trainer_win_rate': trainer_win_rate,
                'weight': weight,
                'recent_form': recent_form,
                'days_since_last_run': days_since_last_run,
                'injury': injury,
                'track': track,  # we'll one-hot encode later
                'weather': weather, # same
                'distance': distance
            }
            horses.append(horse_data)
        
        race_data = {
            'race_id': race_id,
            'race_date': race_date,
            'race_time': race_time,
            'track': track,
            'weather': weather,
            'distance': distance,
            'horses': horses
        }
        races.append(race_data)
    return races
