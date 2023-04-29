from pathlib import Path

import pandas as pd

import streamlit as st
from config import config
from src import main, utils

# Title
st.title("MLOps - Tweet Classification for financial application")

# Data Section
st.header("🔢 Data")
labeled_tweets_fp = Path(config.DATA_DIR, "labeled_tweets.csv")
df = pd.read_csv(labeled_tweets_fp)
st.text(f"Tweets (count: {len(df)})")
st.write(df)


# Performance Section
st.header("📊 Performance")

performance_fp = Path(config.RESULT_DIR, "performance.json")
performance = utils.load_dict(filepath=performance_fp)
st.text("Overall:")
st.json(performance["overall"])

tag = st.selectbox("Choose a tag: ", list(performance["class"].keys()))
st.json(performance["class"][tag])

if "slices" in performance.keys():
    slice = st.selectbox("Choose a slice: ", list(performance["slices"].keys()))
    st.json(performance["slices"][slice])

# Inference Section
st.header("🚀 Inference")

text = st.text_input("Enter text :")
run_id = st.text_input("Enter run ID:", open(Path(config.CONFIG_DIR, "run_id.txt")).read())
prediction = main.predict_tag(text=text, run_id=run_id)
st.json(prediction)
