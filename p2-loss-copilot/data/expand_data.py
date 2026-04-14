"""
expand_data.py
Expands losses.db to 100 rows and rewrites sample_losses.txt with 20 narrative entries.
Safe to run multiple times — wipes synthetic rows and rewrites from scratch.
Keeps existing 8 rows intact (ids 1-8).
Author: Eswar Gutlapalli
"""

import sqlite3
import os
import random

random.seed(42)     # reproducible

DB_PATH = os.path.join(os.path.dirname(__file__), "losses.db")
TXT_PATH = os.path.join(os.path.dirname(__file__), "sample_losses.txt")

# Schema constants (must match existing table exactly)
REGIONS = ["Midwest", "Northeast", "South", "West"]
PRODUCTS = ["Auto Loan", "Commercial RE", "Credit Card",
            "Personal Loan", "Mortgage", "Small Business Loan"]
QUARTERS = ["Q1-2023", "Q2-2023", "Q3-2023", "Q4-2023",
            "Q1-2024", "Q2-2024", "Q3-2024", "Q4-2024"]

# Existing rows already cover these combos — skip to avoid duplication
EXISTING = {
    ("Midwest",   "Auto Loan",     "Q3-2023"),
    ("Northeast", "Commercial RE", "Q3-2023"),
    ("South",     "Credit Card",   "Q2-2023"),
    ("West",      "Auto Loan",     "Q2-2023"),
    ("Midwest",   "Mortgage",      "Q1-2023"),
    ("Northeast", "Credit Card",   "Q3-2023"),
    ("South",     "Commercial RE", "Q1-2023"),
    ("West",      "Mortgage",      "Q2-2023"),
}

# Loss amount ranges by product (in USD, realistic credit loss scale)
LOSS_RANGES = {
    "Auto Loan":          (800_000,  5_500_000),
    "Commercial RE":      (3_000_000, 15_000_000),
    "Credit Card":        (500_000,   4_000_000),
    "Mortgage":           (1_500_000, 8_000_000),
    "Personal Loan":      (300_000,   2_500_000),
    "Small Business Loan":(1_000_000, 6_000_000),
}

# Delinquency ranges by product
DELQ_RANGES = {
    "Auto Loan":          (0.05, 0.16),
    "Commercial RE":      (0.04, 0.12),
    "Credit Card":        (0.10, 0.25),
    "Mortgage":           (0.02, 0.08),
    "Personal Loan":      (0.07, 0.18),
    "Small Business Loan":(0.06, 0.14),
}

def severity(delq: float) -> str:
    if delq >= 0.14:
        return "High"
    elif delq >= 0.07:
        return "Medium"
    return "Low"

def generate_rows(n_target: int = 100) -> list[tuple]:
    """Generate synthetic rows until we reach n_target, skipping existing combos."""
    rows = []
    seen = set(EXISTING)

    attempts = 0
    while len(rows) < (n_target - len(EXISTING)) and attempts < 10_000:
        attempts += 1
        region  = random.choice(REGIONS)
        product = random.choice(PRODUCTS)
        quarter = random.choice(QUARTERS)
        combo   = (region, product, quarter)

        if combo in seen:
            continue
        seen.add(combo)

        lo, hi = LOSS_RANGES[product]
        loss   = random.randint(lo // 100_000, hi // 100_000) * 100_000

        dlo, dhi = DELQ_RANGES[product]
        delq     = round(random.uniform(dlo, dhi), 2)

        rows.append((region, product, quarter, loss, delq, severity(delq)))

    return rows

def expand_db(rows: list[tuple]):
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    # Remove any previously generated synthetic rows (id > 8)
    cur.execute("DELETE FROM losses WHERE id > 8")

    cur.executemany(
        "INSERT INTO losses (region, product, quarter, loss_amount, delinquency_rate, severity) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows
    )
    conn.commit()

    total = cur.execute("SELECT COUNT(*) FROM losses").fetchone()[0]
    conn.close()
    print(f"DB expanded → {total} total rows ({len(rows)} synthetic added)")

NARRATIVES = """Midwest region showed 12% increase in auto loan defaults in Q3-2023, driven by rising unemployment in manufacturing sectors.
Commercial real estate losses concentrated in urban markets of the Northeast, particularly office and retail properties post-pandemic.
Credit card delinquencies rose sharply among the 25-34 age group in Q4-2023 amid elevated consumer debt levels.
Personal loan defaults increased by 8% in the Southern region due to declining disposable income and cost-of-living pressures.
Mortgage delinquencies rose 5% across all regions in Q1-2024 as sustained high interest rates reduced refinancing options.
Small business loan losses concentrated in the retail sector, particularly in the Midwest and South during Q2-2024.
West region auto loan performance improved in Q1-2024 as employment stabilized and used vehicle prices normalized.
Northeast commercial real estate exposure remains elevated with office vacancy rates above 20% in major metro areas.
Credit card charge-offs in the South reached High severity in Q3-2024, correlating with increased revolving balance utilization.
Midwest mortgage delinquency rates remained Low through 2023 due to stable housing demand and conservative underwriting.
Personal loan losses in the West were contained at Medium severity in Q2-2024 despite broader economic headwinds.
Small business loan defaults in the Northeast declined in Q4-2023 following improved access to SBA relief programs.
Auto loan delinquency rates in the South trended upward through 2024, linked to sub-prime originations from 2022.
Commercial real estate recoveries in the West were limited by depressed office property valuations in Q3-2024.
Credit card loss rates moderated in the Midwest in Q1-2024 following aggressive collections activity and balance paydowns.
Mortgage performance in the Northeast deteriorated in Q4-2024 as property tax increases compressed borrower cash flows.
Personal loan delinquencies peaked in Q3-2023 across all regions before stabilizing as consumer spending slowed.
Small business loan losses in the West were concentrated in hospitality and food service sectors through 2023-2024.
Auto loan severity classifications shifted from Low to Medium in the Northeast between Q2-2023 and Q4-2023.
Overall portfolio delinquency trends show Commercial RE and Credit Card as highest-risk products across all regions and periods.""".strip()

def expand_narratives():
    with open(TXT_PATH, "w") as f:
        f.write(NARRATIVES)
    lines = NARRATIVES.split("\n")
    print(f"Narratives rewritten → {len(lines)} entries")

if __name__ == "__main__":
    rows = generate_rows(100)
    expand_db(rows)
    expand_narratives()
    print("Done. Run your app — nothing else changes.")