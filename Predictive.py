import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class PredictiveAnalyticsEngine:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.trained = False

    # =============================
    # LOAD DATA
    # =============================
    def load_data(self, filepath):
        self.data = pd.read_csv(filepath)
        print("Data loaded successfully.")

    # =============================
    # PREPROCESSING
    # =============================
    def preprocess(self, target_column):
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]

        X_scaled = self.scaler.fit_transform(X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        print("Data preprocessed.")

    # =============================
    # TRAIN MODEL
    # =============================
    def train(self):
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        self.trained = True
        print("Model trained.")

    # =============================
    # EVALUATE
    # =============================
    def evaluate(self):
        predictions = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predictions)
        print(f"Mean Squared Error: {mse}")
        return mse

    # =============================
    # PREDICT
    # =============================
    def predict(self, new_data):
        new_data_scaled = self.scaler.transform(new_data)
        return self.model.predict(new_data_scaled)

    # =============================
    # SAVE / LOAD MODEL
    # =============================
    def save_model(self, path="model.pkl"):
        joblib.dump((self.model, self.scaler), path)
        print("Model saved.")

    def load_model(self, path="model.pkl"):
        self.model, self.scaler = joblib.load(path)
        self.trained = True
        print("Model loaded.")
