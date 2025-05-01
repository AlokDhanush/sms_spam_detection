import streamlit as st
import pickle
import numpy as np
import spacy 

nlp = spacy.load("en_core_web_sm")


def text_transformation(text):
    text = text.lower()
    doc = nlp(text) 
    tokens = [token.text for token in doc if not token.is_stop]
    tokens = [token for token in tokens if token.isalnum()]
    tokens = nlp(" ".join(tokens))
    lemmatized_tokens = [token.lemma_ for token in tokens]

    return " ".join(lemmatized_tokens)


with open("spam_ham_model.pkl", "rb") as f:
    model = pickle.load(f)


with open("cv_vectorizer.pkl", "rb") as f:
    cv = pickle.load(f)


st.title("SMS Spam Detection App")

st.markdown(
    """
    Enter an SMS message below and the app will predict whether it's **Spam** or **Ham (Not Spam)**.
    """
)


user_input = st.text_area("Enter SMS message", "")


if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        user_input = text_transformation(user_input)
        user_vec = cv.transform([user_input]) 
        prediction = model.predict(user_vec)[0]
        if prediction == 1:
            st.error("Spam Message Detected!")
        else:
            st.success("This looks like a Ham (Not Spam) message.")
