# model.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

MODEL_FILE = 'horse_racing_model.pkl'

class HorseRacingModel:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        if os.path.exists(MODEL_FILE):
            self.model = joblib.load(MODEL_FILE)
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained with accuracy: {accuracy:.2f}")
        joblib.dump(self.model, MODEL_FILE)
    
    def predict(self, X):
        if self.model is None:
            self.load_model()
        return self.model.predict_proba(X)
    
    def update_model(self, new_data, new_labels):
        # Retrain the model with new data
        self.model.fit(new_data, new_labels)
        joblib.dump(self.model, MODEL_FILE)
        print("Model updated with new data.")

# For simplicity, we assume that the features are:
# ['horse_age', 'jockey_win_rate', 'trainer_win_rate', 'weight', 'distance', ...]
# We'll create a dummy dataset for initial training if no model exists.
