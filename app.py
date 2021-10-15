#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestRegressor


# In[ ]:


st.write(
'''
## Find Out The Sale Price of Your Car
''')


# In[ ]:


# Read model
with open('rf_final_model', 'rb') as read_file:
    rf = pickle.load(read_file)


# In[ ]:


# Load dataframe
df = pd.read_csv('model_ready_df.csv', index_col=0)
features = df[['mpg', 'mileage', 'num_of_entertainment',
               'num_of_safety', 'year', 'drivetrain', 'fuel_type',
               'transmission', 'engine', 'make', 'model']]


# In[ ]:


mpg = st.number_input('Miles per Gallon', value=25)
mileage = st.number_input('Mileage of the Car', value=85600)
entertainment = st.number_input('Number of Entertainment Features', value=1)
safety = st.number_input('Number of Safety Features', value=3)
year = st.number_input('Year The Car Was Made', value=2017)
drivetrain = st.text_input('Type of Drivetrain', value='FWD')
fuel = st.text_input('Type of Fuel Used', value='Gasoline')
transmission = st.text_input('Type of Transmission', value='Automatic')
engine = st.text_input('Type of Engine', value='2.4L')
make = st.text_input('Brand of the Car', value='Jeep')
model = st.text_input('Model of the Car', value='Cherokee')

input_df = pd.DataFrame({'mpg': [mpg], 
                         'mileage': [mileage],
                         'num_of_entertainment': [entertainment],
                         'num_of_safety': [safety],
                         'year': [year],
                         'drivetrain': [drivetrain],
                         'fuel_type': [fuel],
                         'transmission': [transmission],
                         'engine': [engine],
                         'make': [make],
                         'model': [model]}, index=[0])


# In[ ]:


comb_df = pd.concat([features, input_df])
dummified_df = pd.get_dummies(comb_df, drop_first=True)


# In[ ]:


pred = format(rf.predict(np.array(dummified_df.iloc[-1]).reshape(1, -1)))
float_pred = float(pred.strip('[]'))

st.write(
f'Predicted Sale Price of the Car: ${float_pred:.2f}'
)

