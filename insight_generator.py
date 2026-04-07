"""
Financial Insight Generator
Loads loan portfolio data, computes loss rates,
and generates constrained executive risk narratives via Claude API.
Author: Eswar Gutlapalli
"""

import anthropic
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

def generate_insights(df):

    client = anthropic.Anthropic()

    df["loss_rate"] = df["losses_millions"] / df["balance_millions"] * 100

    avg_by_segment = df.groupby("segment")["loss_rate"].mean().round(2)
    q4 = df[df["quarter"] == "Q4"].sort_values("loss_rate", ascending=False)
    total_balance = df["balance_millions"].sum()

    # Step 2: build the prompt from real data ---
    summary = ""
    for seg, rate in avg_by_segment.items():
        summary += f"{seg}: {rate}% average loss rate\n"

    q4_summary = ""
    for _, row in q4.iterrows():
        q4_summary += f"{row["segment"]} : {row["loss_rate"]:.2f}% in Q4\n"

    prompt = f"""
    You are a credit risk analyst. Based on the following loan portfolio data, write a concise 5-sentence executive summary highlighting key risks and trends.

    Total portfolio balance:
    ${total_balance:,.2f}M
    Annual loss rate by segment:
    {summary}
    Q4 loss rates (highest to lowest)
    {q4_summary}

    Focus on which segments need attention and why.

    Important constraints:
    - Use only the numbers explicitly provided in the data above
    - Do not calculate, derive, or infer any additional metrics
    - Do not compute percentage changes, ratios, or comparisons not present in the data
    - If you reference a number, it must appear exactly in the data provided
    """

    message = client.messages.create(
        model = "claude-sonnet-4-6",
        max_tokens = 1024,
        temperature=0.1,     # 0 = deterministic, 1 = creative, default is ~1
        messages = [
            {"role" : "user", "content" : prompt}
        ]
    )

    return message.content[0].text