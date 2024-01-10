
import streamlit as st
import pandas as pd
import numpy as np
import pickle 


st.sidebar.header('User Input Features')

# collects user input features into dataframe

def user_input_features():
    age = st.sidebar.number_input('Enter your age: ')
    
    sex = st.sidebar.selectbox('Sex',(0,1))
    cp = st.sidebar.selectbox('Chest pain type',(0,1))
    trtbps = st.sidebar.number_input('Resting blood preasure: ')
    chol = st.sidebar.number_input('Serum cholesterol in mg/dl: ')
    fbs = st.sidebar.selectbox('Fasting blood sugar',(0,1))
    restech = st.sidebar.number_input('Resting electrocardiographic results: ')
    tha = st.sidebar.number_input('Maximum heart rate achieved: ')
    exng = st.sidebar.selectbox('Exercise induced angina',(0,1))
    old = st.sidebar.number_input('oldpeak: ')
    slp = st.sidebar.number_input('he slope of the peak exercise ST segmen: ')
    ca = st.sidebar.selectbox('number of major vessels',(0,1,2,3))
    thall = st.sidebar.selectbox('thal',(0,1,2))
    
    data = {'age' : age,
            'sex' : sex,
            'cp' : cp,
            'trtbps' : trtbps,
            'chol' : chol,
            'fbs' : fbs,
            'restech' : restech,
            'tha' : tha,
            'exng' : exng,
            'oldpeak' : old,
            'slp': slp,
            'ca' : ca,
            'thall' : thall,
                 }
    
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

#combines user input features with entire dataset
# this will be useful for the encoding phase

heart_dataset = pd.read_csv('Heart Dataset.csv')
heart_dataset = heart_dataset.drop(columns=['output'])

df = pd.concat([input_df,heart_dataset],axis=0)

# encoding of ordinal features
#

df = pd.get_dummies(df, columns = ['sex','cp','fbs','restech','exng','slp','ca','thall'])
 
df = df[:1] # select only the first row(the user inpu data)
 
st.write(input_df)
 
 # reads in saved classification model
 
load_clf = pickle.load(open('KNN.pkl', 'rb'))

# apply model to make prediction
prediction = load_clf.predict(df)
prediction_proba  = load_clf.predict_proba(df)

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probabilitiy')
st.write(prediction_proba)
  
    
    
            
            
            
            
            
            
        
            
            
    



