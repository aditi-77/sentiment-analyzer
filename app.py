import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from transformers import pipeline

st.title("🧠 Smart Sentiment Analyzer")
st.write("Analyze feedback using Hugging Face Transformers!")

# Initialize classifier
@st.cache_resource
def get_classifier():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

classifier = get_classifier()

# Path for feedback storage
DATA_FILE = "feedback.csv"

# Input area
user_text = st.text_area("Enter your feedback:")

# Analyze button
if st.button("Analyze"):
    if user_text.strip():
        result = classifier(user_text)[0]
        sentiment = result['label']
        confidence = result['score']
        st.success(f"Sentiment: {sentiment} ({confidence*100:.2f}% confidence)")
        
        # Save feedback
        df_new = pd.DataFrame([[user_text, sentiment, confidence]],
                              columns=["Text", "Sentiment", "Confidence"])
        if os.path.exists(DATA_FILE):
            df_new.to_csv(DATA_FILE, mode='a', index=False, header=False)
        else:
            df_new.to_csv(DATA_FILE, index=False)
    else:
        st.warning("Please enter some text!")

# Show summary
if st.button("Show Summary"):
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        st.dataframe(df.tail(10))
        st.subheader("Sentiment Distribution")
        sentiment_counts = df["Sentiment"].value_counts()
        fig, ax = plt.subplots()
        sentiment_counts.plot(kind="bar", ax=ax)
        plt.xlabel("Sentiment")
        plt.ylabel("Count")
        st.pyplot(fig)
    else:
        st.warning("No feedback data found!")
