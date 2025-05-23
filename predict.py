import streamlit as st
st.header("Titanic Survival Prediction")
st.subheader("Predicting Survival on the Titanic")
st.image("C:\git\ML\images.jpeg")
import pickle
model=pickle.load(open('model.pkl','rb'))
l_sex=pickle.load(open('l_sex.pkl','rb'))
l_emb=pickle.load(open('l_emb.pkl','rb'))
pclass=st.number_input("Passenger Class")#st.radio('select passenger class',(1,2,3))
sex=st.text_input("Enter Sex:[male,female]")
age=st.number_input("Age")
sibsp=st.number_input("Number of siblings/spouses aboard")
parch=st.number_input("Number of Parents/Children abroad")
fare=st.number_input("Fare")
embarked=st.text_input("Embarked;[S,C,Q]")
if st.button("predict"):
    sex_l=l_sex.transform([sex])[0]
    embarked_l=l_emb.transform([embarked])[0]
    predict=model.predict([[pclass,sex_l,age,sibsp,parch,fare,embarked_l]])[0]
    if (predict==1):
        st.success('Survived')
    else:
        st.warning('Did not Survived')
    
    