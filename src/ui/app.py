"""
Optional Streamlit UI to preview latest report.
Run: streamlit run src/ui/app.py
"""
import os
import json
import streamlit as st

st.set_page_config(page_title="Deal Intelligence", layout="wide")
st.title("Autonomous Deal Intelligence - Dashboard")

out_json = "data/outputs/latest_report.json"
if os.path.exists(out_json):
    with open(out_json, "r") as f:
        data = json.load(f)
    st.subheader("Summary")
    st.write(data.get("summary", ""))
    st.subheader("Findings (raw)")
    st.json(data.get("findings", {}))
else:
    st.info("No report found. Hit the API /run_report to generate one.")
