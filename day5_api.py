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

client = anthropic.Anthropic()

# --- Step 1: load & summarize data ---
df = pd.read_csv('data/loan_data.csv')
df['loss_rate'] = df['losses_millions'] / df['balance_millions'] * 100

avg_by_segment = df.groupby('segment')['loss_rate'].mean().round(2)
q4 = df[df['quarter'] == 'Q4'].sort_values('loss_rate', ascending=False)
total_balance = df['balance_millions'].sum()

# Step 2: build the prompt from real data ---
summary = ""
for seg, rate in avg_by_segment.items():
    summary += f'{seg}: {rate}% average loss rate\n'

q4_summary = ""
for _, row in q4.iterrows():
    q4_summary += f'{row['segment']} : {row['loss_rate']:.2f}% in Q4\n'

prompt = f'''
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
'''

message = client.messages.create(
    model = 'claude-sonnet-4-6',
    max_tokens = 1024,
    temperature=0.1,     # 0 = deterministic, 1 = creative, default is ~1
    messages = [
        {"role" : "user", "content" : prompt}
    ]
)

print("--- Portfolio Credit Risk Summary ---")
print(message.content[0].text)

# Day 5 reflection
# What I learned: How to connect/call to Anthropic LLM, how to build a promt with data, and how to update the prompt for a more complete analysis
# What confused me: iterrows() method and why was it used instead of items()
# One question I have: The same prompt outputs variation in response, is it possibel to keep the narrative same for the same prompt
# Key insight: LLMs will derive metrics beyond provided data
# Solution: explicit constraints in prompt — tell model what NOT to do
# Production rule: always validate any number in AI output against source data

# Notes:
# # items() — for iterating key:value pairs in a Series or dict
# # iterrows() — for iterating rows in a DataFrame
# The difference is the data structure. `avg_by_segment` is a one-dimensional Series — one value per segment. `q4` is a two-dimensional DataFrame — multiple columns per row. You need `iterrows()` when you need to access multiple columns from the same row, like `row['segment']` and `row['loss_rate']` simultaneously.
# Temperature controls how creative vs deterministic the model's responses are. Higher temperature = more variation. Lower = more consistent
# For a risk reporting tool, low temperature is correct — you want reliability, not creativity. For a brainstorming tool, you'd want higher temperature.
# Important about how LLMs work — even at low temperature, they're not deterministic like a calculator. They're probabilistic. You can constrain the variation but not eliminate it entirely. 
