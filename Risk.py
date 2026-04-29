class RiskDetectionModel:
    def __init__(self):
        # Define weights for each risk factor
        self.weights = {
            "failed_logins": 2.0,
            "transaction_amount": 0.01,
            "location_mismatch": 5.0,
            "unusual_activity": 3.0
        }

    def calculate_risk_score(self, data):
        score = 0

        score += data.get("failed_logins", 0) * self.weights["failed_logins"]
        score += data.get("transaction_amount", 0) * self.weights["transaction_amount"]
        score += (1 if data.get("location_mismatch") else 0) * self.weights["location_mismatch"]
        score += (1 if data.get("unusual_activity") else 0) * self.weights["unusual_activity"]

        return score

    def classify_risk(self, score):
        if score < 20:
            return "LOW"
        elif score < 50:
            return "MEDIUM"
        else:
            return "HIGH"

    def evaluate(self, data):
        score = self.calculate_risk_score(data)
        level = self.classify_risk(score)
        return {
            "risk_score": score,
            "risk_level": level
        }


# Example usage
if __name__ == "__main__":
    model = RiskDetectionModel()

    sample_data = {
        "failed_logins": 3,
        "transaction_amount": 1200,
        "location_mismatch": True,
        "unusual_activity": True
    }

    result = model.evaluate(sample_data)
    print(result)
