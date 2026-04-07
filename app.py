"""
app.py
Streamlit UI for the Financial Insight Generator.
Accepts financial data input and returns Claude-powered insights.
"""
import streamlit as st
import pandas as pd
from insight_generator import generate_insights

st.set_page_config(page_title = "Financial Insight Generator", layout="wide")

st.title("Financial Insight Generator")
st.caption("Powered by Claude. Upload your data and get instant insights")

# File upload
upload_file = st.file_uploader("Upload financial CSV", type=["csv"])

if upload_file:
    df = pd.read_csv(upload_file)
    st.subheader("Preview")
    st.dataframe(df.head())

    # Generate button
    if st.button("Generate Insights"):
        with st.spinner("Analyzing with Claude..."):
            insights = generate_insights(df)

            st.subheader("Insights")
            st.write(insights)
