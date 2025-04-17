import pickle
import streamlit as st
import pandas as pd
import numpy as np

#upload Data
Data = pickle.load(open("Zomato_resturant_yousefeldardery.sav", 'rb'))

st.title("Zomato Restaurant Recommendation System")
st.header("Find the best restaurant for you!")
st.subheader("Enter your preferences below:")


st.sidebar.header("restaurant details")
st.sidebar.text("Please fill in the following details to get a restaurant recommendation.")
st.sidebar.text("Note: All fields are mandatory.")
online_order =st.selectbox("online_order",['Yes', 'No'])

book_table = st.selectbox("book_table",['Yes', 'No'])
location   = st.selectbox('location',['Banashankari', 'Basavanagudi', 'other', 'Jayanagar', 'JP Nagar',
       'Bannerghatta Road', 'BTM', 'Electronic City', 'Wilson Garden',
       'Shanti Nagar', 'Koramangala 5th Block', 'Richmond Road', 'HSR',
       'Koramangala 7th Block', 'Bellandur', 'Sarjapur Road',
       'Marathahalli', 'Whitefield', 'Old Airport Road', 'Indiranagar',
       'Koramangala 1st Block', 'Frazer Town', 'MG Road', 'Brigade Road',
       'Lavelle Road', 'Church Street', 'Ulsoor', 'Residency Road',
       'Shivajinagar', 'St. Marks Road', 'Cunningham Road',
       'Commercial Street', 'Vasanth Nagar', 'Domlur',
       'Koramangala 8th Block', 'Ejipura', 'Jeevan Bhima Nagar',
       'Kammanahalli', 'Koramangala 6th Block', 'Brookefield',
       'Koramangala 3rd Block', 'Koramangala 4th Block', 'Banaswadi',
       'Kalyan Nagar', 'Malleshwaram', 'Rajajinagar', 'New BEL Road'])

Restaurant_type = st.selectbox('rest_type',[1, 2])

Cuisines =  st.selectbox('cuisines',[3, 2, 1, 4, 5, 8, 7, 6])

approx_cost =  st.slider('approx_cost',min_value=30,max_value=6000,value=0,step=1)

listed_in_type = st.selectbox('listed_in_type',['Buffet', 'Cafes', 'Delivery', 'Desserts', 'Dine-out',
       'Drinks & nightlife', 'Pubs and bars'])

listed_in_city=st.selectbox('listed_in_city',['Banashankari', 'Bannerghatta Road', 'Basavanagudi', 'Bellandur',
       'Brigade Road', 'Brookefield', 'BTM', 'Church Street',
       'Electronic City', 'Frazer Town', 'HSR', 'Indiranagar',
       'Jayanagar', 'JP Nagar', 'Kalyan Nagar', 'Kammanahalli',
       'Koramangala 4th Block', 'Koramangala 5th Block',
       'Koramangala 6th Block', 'Koramangala 7th Block', 'Lavelle Road',
       'Malleshwaram', 'Marathahalli', 'MG Road', 'New BEL Road',
       'Old Airport Road', 'Rajajinagar', 'Residency Road',
       'Sarjapur Road', 'Whitefield'])


data_frame_deploy=pd.DataFrame({
    'online_order'   : [online_order],
    'book_table'     : [book_table],
    'location'       : [location],
    'rest_type'      : [Restaurant_type],
    'cuisines'       : [Cuisines],
    'approx_cost'    : [approx_cost],
    'listed_in(type)': [listed_in_type],
    'listed_in(city)': [listed_in_city]
    },index=[0]
)

Confirm = st.sidebar.button("Click to predict")

if Confirm:
    result=Data.predict(data_frame_deploy)
    # st.sidebar.write(f"The restaurant recommendation is:{result}")
    if result[0] == 1:
        st.sidebar.success("Success!")
        st.sidebar.image("https://miro.medium.com/v2/resize:fit:1100/format:webp/1*vNVomBKGpJZEKI4PrQT5hQ.jpeg", width=300)
    else:
        st.sidebar.error("Fail!")
        st.sidebar.image("https://testteach.co.uk/wp-content/uploads/2021/01/AdobeStock_275170280-1.png", width=300)
