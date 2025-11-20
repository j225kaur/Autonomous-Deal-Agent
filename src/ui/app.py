"""
Optional Streamlit UI to preview latest report.
Run: streamlit run src/ui/app.py
"""
import os
import json
import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Deal Intelligence", layout="wide")
st.title("Autonomous Deal Intelligence - Dashboard")

out_json = "data/outputs/latest_report.json"

@st.cache_data(ttl=60)
def load_data():
    if os.path.exists(out_json):
        with open(out_json, "r") as f:
            return json.load(f)
    return None

data = load_data()

if not data:
    st.info("No report found. Hit the API /run_report to generate one.")
    st.stop()

findings = data.get("findings", {})
deals = findings.get("deals", [])
signal_scores = findings.get("signal_scores", {})
summary = data.get("summary", "")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Signal Analysis", "Retrieved Docs", "Timeline"])

with tab1:
    st.subheader("Executive Summary")
    st.write(summary)
    
    st.subheader("Identified Deals")
    if deals:
        df_deals = pd.DataFrame(deals)
        st.dataframe(df_deals, use_container_width=True)
    else:
        st.write("No definitive deals identified.")

with tab2:
    st.subheader("Signal Scores")
    if signal_scores:
        # Create a dataframe for scores
        rows = []
        for ticker, info in signal_scores.items():
            rows.append({
                "Ticker": ticker,
                "Score": info.get("score", 0.0),
                "Explanation": ", ".join(info.get("explanation", []))
            })
        df_scores = pd.DataFrame(rows)
        
        # Bar chart
        chart = alt.Chart(df_scores).mark_bar().encode(
            x='Ticker',
            y='Score',
            color=alt.Color('Score', scale=alt.Scale(scheme='redyellowgreen')),
            tooltip=['Ticker', 'Score', 'Explanation']
        ).properties(title="Deal Probability Score")
        
        st.altair_chart(chart, use_container_width=True)
        
        st.subheader("Detailed Breakdown")
        for ticker, info in signal_scores.items():
            with st.expander(f"{ticker} (Score: {info.get('score', 0.0):.2f})"):
                st.write(f"**Explanation:** {', '.join(info.get('explanation', []))}")
                st.write("**Components:**")
                st.json(info.get("components", {}))
    else:
        st.write("No signal scores available.")

with tab3:
    st.subheader("Retrieved Evidence")
    # We need to access retrieved docs. 
    # If they are not in latest_report.json, we might need to adjust ReportGenerator to include them.
    # For now, let's check if they are in findings or we can infer.
    # Actually ReportGenerator puts findings in json.
    # Let's assume we might want to see what was retrieved.
    # If not available, we show a placeholder.
    st.write("Raw findings data:")
    st.json(findings)

with tab4:
    st.subheader("Timeline")
    st.write("Timeline visualization coming soon.")
    # If we had dates in deals, we could plot a timeline.
    if deals:
        st.write("Deal Events:")
        for d in deals:
            st.write(f"- {d.get('status', 'Unknown')}: {d.get('acquirer')} -> {d.get('target')}")
