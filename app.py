import streamlit as st
import pickle


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))
transformText = pickle.load(open('transformText.pkl','rb'))

st.title("SMS Spam Classifier")

inputSMS= st.text_input("Enter the message")


# 1. preprocess
transformedSMS = transformText(inputSMS)
# 2. vectorize
vector_input = tfidf.transform([transformedSMS])
# 3. predict
result = model.predict(vector_input)[0]
# 4. display
if result == 1:
    st.header("Spam")
else:
    st.header("Not Spam")