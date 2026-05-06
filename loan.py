from dataclasses import dataclass
from typing import Dict, Any


# ----------------------------
# Loan Application Input
# ----------------------------
@dataclass
class LoanApplication:
    income: float               # monthly income
    credit_score: int           # 300 - 850
    debt: float                 # monthly debt
    loan_amount: float          # requested loan amount
    employment_years: float     # job stability
    late_payments: int          # history indicator


# ----------------------------
# Decision Engine
# ----------------------------
class LoanDecisionEngine:

    def debt_to_income(self, app: LoanApplication):
        return app.debt / app.income if app.income > 0 else 1

    # ----------------------------
    # 1. Hard Rule Filters (Instant Rejects)
    # ----------------------------
    def rule_checks(self, app: LoanApplication):
        if app.credit_score < 500:
            return "REJECT: Credit score too low"

        if self.debt_to_income(app) > 0.6:
            return "REJECT: Debt-to-income too high"

        if app.late_payments >= 5:
            return "REJECT: Excessive late payments"

        return None

    # ----------------------------
    # 2. Risk Scoring Model
    # ----------------------------
    def risk_score(self, app: LoanApplication):
        score = 0

        # Credit score contribution
        score += (app.credit_score - 300) / 550 * 40

        # Income strength
        score += min(app.income / app.loan_amount, 2) * 20

        # Employment stability
        score += min(app.employment_years, 10) * 2

        # Debt penalty
        score -= self.debt_to_income(app) * 50

        # Late payment penalty
        score -= app.late_payments * 5

        return score

    # ----------------------------
    # 3. Final Decision Logic
    # ----------------------------
    def decide(self, app: LoanApplication):

        # Step 1: rule-based rejection
        rule_result = self.rule_checks(app)
        if rule_result:
            return {
                "decision": "REJECT",
                "reason": rule_result
            }

        # Step 2: compute risk score
        score = self.risk_score(app)

        # Step 3: decision thresholds
        if score >= 60:
            decision = "APPROVE"
        elif 40 <= score < 60:
            decision = "REVIEW"
        else:
            decision = "REJECT"

        return {
            "decision": decision,
            "risk_score": round(score, 2),
            "debt_to_income": round(self.debt_to_income(app), 2)
        }


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == "__main__":
    engine = LoanDecisionEngine()

    applicant = LoanApplication(
        income=5000,
        credit_score=680,
        debt=1200,
        loan_amount=10000,
        employment_years=3,
        late_payments=1
    )

    result = engine.decide(applicant)
    print(result)
