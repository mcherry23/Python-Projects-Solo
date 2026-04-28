import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

class AIModelBuilder:
    def __init__(self):
        self.model = None
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    # =========================
    # DATA LOADING
    # =========================
    def load_data(self, filepath, target_column):
        self.data = pd.read_csv(filepath)
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print("Data loaded successfully!")

    # =========================
    # MODEL SELECTION
    # =========================
    def select_model(self, model_type="classification", model_name="random_forest"):
        if model_type == "classification":
            if model_name == "logistic_regression":
                self.model = LogisticRegression(max_iter=1000)
            elif model_name == "random_forest":
                self.model = RandomForestClassifier()

        elif model_type == "regression":
            if model_name == "linear_regression":
                self.model = LinearRegression()
            elif model_name == "random_forest":
                self.model = RandomForestRegressor()

        print(f"Model selected: {self.model}")

    # =========================
    # TRAIN MODEL
    # =========================
    def train(self):
        if self.model is None:
            raise Exception("No model selected")
        self.model.fit(self.X_train, self.y_train)
        print("Model training complete")

    # =========================
    # EVALUATE MODEL
    # =========================
    def evaluate(self, task_type="classification"):
        predictions = self.model.predict(self.X_test)

        if task_type == "classification":
            score = accuracy_score(self.y_test, predictions)
            print("Accuracy:", score)
        else:
            mse = mean_squared_error(self.y_test, predictions)
            print("MSE:", mse)

    # =========================
    # SAVE / LOAD MODEL
    # =========================
    def save_model(self, filename="model.pkl"):
        joblib.dump(self.model, filename)
        print("Model saved")

    def load_model(self, filename="model.pkl"):
        self.model = joblib.load(filename)
        print("Model loaded")


# =========================
# SIMPLE CLI INTERFACE
# =========================
if __name__ == "__main__":
    builder = AIModelBuilder()

    print("=== AI MODEL BUILDER ===")
    filepath = input("Enter CSV file path: ")
    target = input("Enter target column name: ")

    builder.load_data(filepath, target)

    task = input("Task type (classification/regression): ")
    model = input("Model (logistic_regression/random_forest/linear_regression): ")

    builder.select_model(task, model)
    builder.train()
    builder.evaluate(task)

    save = input("Save model? (y/n): ")
    if save.lower() == "y":
        builder.save_model()

