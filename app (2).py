
import streamlit as st
import pickle
# Save Logistic Regression model
with open("logistic_model.pkl", "wb") as f:
    pickle.dump(lr_model, f)

# Save Naive Bayes model
with open("naive_bayes_model.pkl", "wb") as f:
    pickle.dump(nb_model, f)

# Save TF-IDF Vectorizer
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

# Save Label Encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)



# Title and instructions
st.title("📝 Sentiment Analysis App")
st.markdown("Enter a product review and choose a model to classify it as **Positive**, **Negative**, or **Neutral**.")

# User input
review = st.text_area("✏️ Enter your review below:")

# Model selection
model_choice = st.selectbox("🔍 Choose a model", ["Logistic Regression", "Naive Bayes"])

# Predict button
if st.button("Predict"):
    if review.strip() == "":
        st.warning("⚠️ Please enter a review to analyze.")
    else:
        # Preprocess input
        vector_input = tfidf_vectorizer.transform([review])

        # Make prediction
        if model_choice == "Logistic Regression":
            prediction = logistic_model.predict(vector_input)[0]
        else:
            prediction = naive_bayes_model.predict(vector_input)[0]

        # Decode label
        sentiment = label_encoder.inverse_transform([prediction])[0]
        st.success(f"✅ Predicted Sentiment: **{sentiment.upper()}**")
