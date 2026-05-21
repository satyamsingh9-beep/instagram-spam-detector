import streamlit as st
import pandas as pd
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Paths
base_dir = Path(__file__).resolve().parent.parent
csv_path = base_dir / "data" / "Youtube01-Psy.csv"

st.title("Instagram Spam Detector")
st.write("Detect whether a comment is spam or genuine.")

try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    st.error(f"Dataset not found at: {csv_path}")
    st.stop()

if "CONTENT" not in df.columns or "CLASS" not in df.columns:
    st.error("Dataset is missing required columns: CONTENT and CLASS.")
    st.stop()

df = df[["CONTENT", "CLASS"]]

# Clean text
def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^a-zA-Z ]", "", text)

    return text

# Apply cleaning
df['CONTENT'] = df['CONTENT'].apply(clean_text)

# Vectorization
vectorizer = TfidfVectorizer(ngram_range=(1,2))

X = vectorizer.fit_transform(df['CONTENT'])

y = df['CLASS']

# Train model
model = LogisticRegression()

model.fit(X, y)

comment = st.text_input("Enter Instagram Comment")

if st.button("Check Comment"):
    if not comment:
        st.warning("Please enter a comment to analyze.")
    else:
        cleaned = clean_text(comment)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)

        if prediction[0] == 1:
            st.error("🚨 Spam Comment")
        else:
            st.success("✅ Genuine Comment")