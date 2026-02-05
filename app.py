import streamlit as st
import pickle

model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📧 Spam Email Detection")
st.write("Enter an email message to check if it is **Spam or Not Spam**")

email_text = st.text_area("Email Content")

if st.button("Check"):
    if email_text.strip() == "":
        st.warning("Please enter some text")
    else:
        email_vec = vectorizer.transform([email_text])
        prediction = model.predict(email_vec)

        if prediction[0] == 1:
            st.error("🚨 This is a SPAM Email")
        else:
            st.success("✅ This is NOT a Spam Email")
