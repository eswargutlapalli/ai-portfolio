"""
create_db.py
Seeds losses.db with sample structured loss data.
Author: Eswar Gutlapalli
"""

import pandas as pd
import sqlite3

def create_losses_db(db_path: str = "data/losses.db"):
    conn = sqlite3.connect(db_path)

    losses = pd.DataFrame([
        {"id": 1, "region": "Midwest",   "product": "Auto Loan",    "quarter": "Q3-2023", "loss_amount": 4_200_000, "delinquency_rate": 0.12, "severity": "High"},
        {"id": 2, "region": "Northeast", "product": "Commercial RE", "quarter": "Q3-2023", "loss_amount": 9_800_000, "delinquency_rate": 0.08, "severity": "High"},
        {"id": 3, "region": "South",     "product": "Credit Card",   "quarter": "Q2-2023", "loss_amount": 1_500_000, "delinquency_rate": 0.15, "severity": "Medium"},
        {"id": 4, "region": "West",      "product": "Auto Loan",     "quarter": "Q2-2023", "loss_amount": 2_100_000, "delinquency_rate": 0.06, "severity": "Low"},
        {"id": 5, "region": "Midwest",   "product": "Mortgage",      "quarter": "Q1-2023", "loss_amount": 3_300_000, "delinquency_rate": 0.04, "severity": "Low"},
        {"id": 6, "region": "Northeast", "product": "Credit Card",   "quarter": "Q3-2023", "loss_amount": 2_700_000, "delinquency_rate": 0.18, "severity": "High"},
        {"id": 7, "region": "South",     "product": "Commercial RE", "quarter": "Q1-2023", "loss_amount": 5_600_000, "delinquency_rate": 0.09, "severity": "Medium"},
        {"id": 8, "region": "West",      "product": "Mortgage",      "quarter": "Q2-2023", "loss_amount": 4_400_000, "delinquency_rate": 0.05, "severity": "Low"},
    ])

    losses.to_sql("losses", conn, if_exists="replace", index=False)
    conn.close()
    print(f"losses.db created at {db_path} with {len(losses)} rows")

if __name__ == "__main__":
    create_losses_db()
