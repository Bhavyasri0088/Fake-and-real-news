import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model and vectorizer
try:
    with open(r"C:\Users\User\Downloads\model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open(r"C:\Users\User\Downloads\vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
except Exception as e:
    st.error(f"Error loading model/vectorizer: {e}")
    st.stop()

# Streamlit UI
def main():
    st.title("📰 Fake News Detection App")
    st.subheader("Detect whether a news article is Fake or Real")
    
    # Input from user
    user_input = st.text_area("Enter news text below:", height=200)
    
    if st.button("Check News Authenticity"):
        if user_input.strip() == "":
            st.warning("Please enter news text to analyze.")
        else:
            prediction = predict_fake_news(user_input)
            if prediction == 1:
                st.error("⚠️ Fake News Detected!")
            else:
                st.success("✅ The news appears to be Real.")

# Prediction Function
def predict_fake_news(text):
    try:
        transformed_text = vectorizer.transform([text])  # Convert text to TF-IDF features
        prediction = model.predict(transformed_text)[0]  # Predict
        return prediction
    except Exception as e:
        st.error(f"Error processing input: {e}")
        return None

if __name__ == "__main__":
       main()


   

