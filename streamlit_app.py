import streamlit as st 
import pickle
import pandas as pd
import sklearn

model = pickle.load(open('ml_model.pkl', 'rb'))

# Set the app title 
st.title('Sentiment Analysis App') 
# Add a welcome message 
st.write('Type message and enter to see the result!') 
# Create a text input 
user_input = st.text_input('Enter a message:', 'Your message :)') 
st.write('-------------------------------')
prediction = model.predict([user_input])
if prediction == "pos":
  pred_msg = "POSITIVE 😀"
elif prediction == "neg":
  pred_msg = "NEGATIVE 😖"
else:
  pred_msg = "NEUTRAL"

# Display the customized message 
st.write(f"Your Message: **{user_input}**")
st.write('-------------------------------')
st.write(f"Prediction: **{pred_msg}**")

