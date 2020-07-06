#!streamlit run app.py

import numpy as np
import streamlit as st 
import pickle
import pandas as pd
from PIL import Image

pickle_in = open("models/Haberman_model.sav","rb")
classifier=pickle.load(pickle_in)
sclr=classifier[1]
classifier=classifier[0]

def predictSurvivalStat(Age,Operation_Year,No_Of_Lymph_Nodes):
    
    """To predict Whether a patient will survive 5 years or more after the operation based on age , year of operation and the number of positive axillary nodes
    ---
    parameters:  
      - name: Age
        in: query
        type: number
        required: true
      - name: Operation_Year
        in: query
        type: number
        required: true
      - name: No_Of_Lymph_Nodes
        in: query
        type: number
        required: true
    responses:
        200:
            description: The Survival Status :
        
    """
    normal= [[Age,Operation_Year,No_Of_Lymph_Nodes]]
    normal=sclr.transform(normal)
    prediction=classifier.predict(normal)
   #print(prediction[0])
    return prediction



def main():
    st.title("")
    html_temp = """
    <div style="background-color:#33FFBD;padding:10px">
    <h2 style="color:white;text-align:center;">HABERMAN CANCER_SURVIVAL APP </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Age = st.text_input("Age","Type Here")
    Operation_Year = st.text_input("Operation_Year","Type Here")
    No_Of_Lymph_Nodes = st.text_input("No_Of_Lymph_Nodes","Type Here")
    result=""
    if st.button("Predict"):
        result=predictSurvivalStat(Age,Operation_Year,No_Of_Lymph_Nodes)[0]
        st.success('The Person is Likely to {}'.format("Survive." if result==1 else "Die."))
    if st.button("About"):
        st.text("HABERMAN_DATASET")
        

if __name__=='__main__':
    main()
    
    
    
