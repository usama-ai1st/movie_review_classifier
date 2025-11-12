import streamlit as st
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model

# Load model
model = load_model("movie_review_classifier.h5")

import json

with open("imdb_word_index.json", "r") as f:
    word_index = json.load(f)

# Reverse mapping
word_to_id = {k: (v + 3) for k, v in word_index.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2
word_to_id["<UNUSED>"] = 3

# Vectorization function
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            if j < dimension:
                results[i, j] = 1.0
    return results

# Streamlit UI
st.set_page_config(page_title="Movie Review Sentiment Classifier", page_icon="🎬", layout="centered")

st.title("🎥 Movie Review Sentiment Classifier")
st.write("Enter a movie review below to predict whether it's **Positive** or **Negative**!")

review = st.text_area("Your Review", placeholder="Type your movie review here...")

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review before prediction.")
    else:
        # Tokenize and vectorize
        encoded = [word_to_id.get(word.lower(), 2) for word in review.split()]
        x_custom = vectorize_sequences([encoded])

        # Predict
        prediction = model.predict(x_custom)
        score = prediction[0][0]
        confidence = score * 100 if score >= 0.5 else (1 - score) * 100

        # Display result
        sentiment = "Positive 😊" if score >= 0.5 else "Negative 😞"
        st.subheader(f"**Sentiment:** {sentiment}")
        st.metric(label="Model Confidence", value=f"{confidence:.2f}%")

        # Optional: progress bar for fun
        st.progress(int(confidence))

# Footer
st.markdown("---")
st.caption("Built with TensorFlow & Streamlit by Usama")
