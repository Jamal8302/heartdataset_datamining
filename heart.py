import streamlit as st
import pandas as pd 
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns 
import pickle 

#import model 
knn = pickle.load(open('KNN.pkl','rb'))

#load dataset
data = pd.read_csv('Heart Dataset.csv')

#data = data.drop(data.columns[0],axis=1)

st.title('Aplikasi Heart Dataset')

html_layout1 = """
<br>
<div style="background-color:red ; padding:2px">
<h2 style="color:white;text-align:center;font-size:35px"><b>Heart Diases Checkup</b></h2>
</div>
<br>
<br>
"""
st.markdown(html_layout1,unsafe_allow_html=True)
activities = ['KNN','Model Lain']
option = st.sidebar.selectbox('Pilihan mu ?',activities)
st.sidebar.header('Data Pasien')

if st.checkbox("Tentang Dataset"):
    html_layout2 ="""
    <br>
    <p>Ini adalah Heart Dataset</p>
    """
    st.markdown(html_layout2,unsafe_allow_html=True)
    st.subheader('Dataset')
    st.write(data.head(10))
    st.subheader('Describe dataset')
    st.write(data.describe())

sns.set_style('darkgrid')

if st.checkbox('EDa'):
    pr =ProfileReport(data,explorative=True)
    st.header('**Input Dataframe**')
    st.write(data)
    st.write('---')
    st.header('**Profiling Report**')
    st_profile_report(pr)

#train test split
X = data.drop('output',axis=1)
y = data['output']
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

#Training Data
if st.checkbox('Train-Test Dataset'):
    st.subheader('X_train')
    st.write(X_train.head())
    st.write(X_train.shape)
    st.subheader("y_train")
    st.write(y_train.head())
    st.write(y_train.shape)
    st.subheader('X_test')
    st.write(X_test.shape)
    st.subheader('y_test')
    st.write(y_test.head())
    st.write(y_test.shape)

def user_report():
    age=st.selectbox ("Age",range(1,121,1))
    sex = st.radio("Select Gender: ", ('male', 'female'))
    cp = st.selectbox('Chest Pain Type',("Typical angina","Atypical angina","Non-anginal pain","Asymptomatic")) 
    trtbps=st.selectbox('Resting Blood Sugar',range(1,500,1))
    restecg=st.selectbox('Resting Electrocardiographic Results',("Nothing to note","ST-T Wave abnormality","Possible or definite left ventricular hypertrophy"))
    chol=st.selectbox('Serum Cholestoral in mg/dl',range(1,1000,1))
    fbs=st.radio("Fasting Blood Sugar higher than 120 mg/dl", ['Yes','No'])
    thalachh=st.selectbox('Maximum Heart Rate Achieved',range(1,300,1))
    exng=st.selectbox('Exercise Induced Angina',["Yes","No"])
    oldpeak=st.number_input('Oldpeak')
    slp = st.selectbox('Heart Rate Slope',("Upsloping: better heart rate with excercise(uncommon)","Flatsloping: minimal change(typical healthy heart)","Downsloping: signs of unhealthy heart"))
    caa=st.selectbox('Number of Major Vessels Colored by Flourosopy',range(0,5,1))
    thall=st.selectbox('Thalium Stress Result',range(1,8,1))
    
    user_report_data = {
        'age':age,
        'sex':sex,
        'cp':cp,
        'trtbps':trtbps,
        'chol':chol,
        'fbs':fbs,
        'restecg':restecg,
        'thalachh':thalachh,
        'exng' : exng,
        'oldpeak' : oldpeak,
        'slp': slp,
        'caa' : caa,
        'thall' : thall,
        
    }
    report_data = pd.DataFrame(user_report_data,index=[0])
    return report_data

#Data Pasion
user_data = user_report()
st.subheader('Data Pasien')
st.write(user_data)

user_result = knn.predict(user_data)
knn_score = accuracy_score(y_test,knn.predict(X_test))

#output
st.subheader('Hasilnya adalah : ')
output=''
if user_result[0]==0:
    output='Kamu Aman'
else:
    output ='Kamu terkena heart diases'
st.title(output)
st.subheader('Model yang digunakan : \n'+option)
st.subheader('Accuracy : ')
st.write(str(knn_score*100)+'%')


