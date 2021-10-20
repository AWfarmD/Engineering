#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import time, os
import numpy as np
import pickle

from bs4 import BeautifulSoup
from sqlalchemy import create_engine

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0',
}


# In[ ]:


def get_car_urls(page_url):
    '''
    Get all the cars urls on one page
    '''
    
    response = requests.get(page_url, headers=headers)
    page = response.text
    soup = BeautifulSoup(page, 'lxml')
    number=len(soup.find_all('a', class_='vehicle-card-link js-gallery-click-link'))
    
    urls = []
    for i in range(number):
        url = soup.find_all('a', class_='vehicle-card-link js-gallery-click-link')[i]['href']
        urls.append(f'http://www.cars.com/{url}')
    return urls


# In[ ]:


def get_car_city_urls(page, zipcodes):
    '''
    Get all urls for a city
    '''
    
    urls = []
    for i in range(page):
        for zipcode in zipcodes:
            base = 'https://www.cars.com/shopping/results/?list_price_max=&makes[]=&maximum_distance=50&models[]=&page={}&page_size=100&stock_type=used&zip={}'.format(i+1, zipcode)
            urls.append(get_car_urls(base))
        
    city_urls = []
    for sub in urls:
        for j in sub:
            city_urls.append(j)
    return city_urls


# In[ ]:


def get_features(page, zipcodes):
    '''
    Get car info into dataframe
    '''
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0',
     }
    
    cars = {}
    
    i=0
    for url in get_car_city_urls(page, zipcodes):
        response = requests.get(url, headers=headers)
        page = response.text
        soup = BeautifulSoup(page, 'lxml')
        
        try:
            name = soup.find(class_='listing-title').text
        except (AttributeError, IndexError, ValueError) as e:
            name = np.nan
            
        try:
            price = int(''.join(soup.find(class_='primary-price').text.strip('$').split(',')))
        except (AttributeError, IndexError, ValueError) as e:
            price = np.nan
            
        try:
            mpg = (int(soup.find(class_='sds-tooltip').find('span').text.split('–')[0]) + int(soup.find(class_='sds-tooltip').find('span').text.split('–')[1]))/2
        except (AttributeError, IndexError, ValueError) as e:
            mpg = np.nan
        
        try:
            mi = int(''.join(soup.find(class_='listing-mileage').text.strip(' .mi').split(',')))
        except (AttributeError, IndexError, ValueError) as e:
            mi = np.nan
        
        try:
            dt = soup.find(class_='fancy-description-list').find_all('dd')[2].text.strip(' ')
        except (AttributeError, IndexError, ValueError) as e:
            dt = np.nan
            
        try:
            fuel = soup.find(class_='fancy-description-list').find_all('dd')[4].text.strip(' ')
        except (AttributeError, IndexError, ValueError) as e:
            fuel = np.nan
        
        try:
            trans = soup.find(class_='fancy-description-list').find_all('dd')[5].text
        except (AttributeError, IndexError, ValueError) as e:
            trans = np.nan
        
        try:
            engine = soup.find(class_='fancy-description-list').find_all('dd')[6].text
        except (AttributeError, IndexError, ValueError) as e:
            engine = np.nan
            
        try:
            ent = len(soup.find_all(class_='vehicle-features-list')[1].find_all('li'))
        except (AttributeError, IndexError, ValueError) as e:
            ent = np.nan
            
        try:
            safe = len(soup.find_all(class_='vehicle-features-list')[3].find_all('li'))
        except (AttributeError, IndexError, ValueError) as e:
            try:
                safe = len(soup.find_all(class_='vehicle-features-list')[2].find_all('li'))
            except (AttributeError, IndexError, ValueError) as e:
                safe = np.nan
        
        cars[name] = [price, mpg, mi, dt, fuel, trans, engine, ent, safe]
    
        cars_df = pd.DataFrame(cars).T
        cars_df.columns = ['price','mpg', 'mileage', 'drivetrain', 'fuel_type', 
                           'transmission', 'engine', 'num_of_entertainment',
                           'num_of_safety']
        i += 1
        if i % 100 == 0:
            print(i)
        
    return cars_df


# In[ ]:


def complete_make(kind):
    '''
    Create full name of the model
    '''
    
    if 'land' in kind.lower():
        kind = 'Land Rover'
    if 'alfa' in kind.lower():
        kind = 'Alfa Romeo'
    return kind


# In[ ]:


def simplify_drivetrain(dt):
    '''
    Simplify drivetrain to be FWD, AWD, 4WD, or RWD
    '''
    
    if dt == 'Front-wheel Drive' or dt == 'Front-Wheel Drive' or dt == 'Front Wheel Drive':
        dt = 'FWD'
    elif dt == 'All-wheel Drive' or dt == 'All Wheel Drive':
        dt = 'AWD'
    elif dt == 'Four-wheel Drive' or dt == 'Four Wheel Drive':
        dt = '4WD'
    elif dt == 'Rear-wheel Drive' or dt == 'Rear Wheel Drive':
        dt = 'RWD'
        
    return dt


# In[ ]:


def simplify_fuel_type(kind):
    '''
    Simplify drivetrain to be gasoline, flex, hybrid, diesel, electric,
    natrual gas, or hydrogen
    '''
    
    if 'gasoline' in kind.lower():
        kind = 'Gasoline'
    elif 'flex' in kind.lower():
        kind = 'Flex'
    elif 'hybrid' in kind.lower():
        kind = 'Hybrid'
    elif 'diesel' in kind.lower():
        kind = 'Diesel'
    elif 'hydrogen' in kind.lower():
        kind = 'Hydrogen'
    elif 'natural' in kind.lower():
        kind = 'NaturalGas'
    return kind


# In[ ]:


def simplify_transmission(kind):
    '''
    Simplify transmission to automatic, manual, or unknown
    '''
    
    if 'automatic' in kind.lower():
        kind = 'Automatic'
    elif 'manual' in kind.lower() or 'm/t' in kind.lower() or 'clutch' in kind.lower():
        kind = 'Manual'
    elif 'a/t' in kind.lower() or 'variable' in kind.lower() or 'continuously' in kind.lower() or 'auto' in kind.lower() or 'cvt' in kind.lower() or 'doppelkupplung' in kind.lower() or 'ivt' in kind.lower() or 'shiftronic' in kind.lower() or 'manumatic' in kind.lower() or 'dsg' in kind.lower() or 'at' in kind.lower() or 'tronic' in kind.lower() or 'a' in kind.lower() or 'vvt' in kind.lower():
        kind = 'Automatic'
    else:
        kind = 'unknown'
    return kind


# In[ ]:


def simplify_engine(kind):
    '''
    Simplify engine to '''
    if '.' in kind:
        kind = kind[kind.find('.')-1:kind.find('.')+2] + 'L'
    else:
        kind = 'Unknown'    
    return kind


# In[ ]:


def pipeline(page, zipcodes):
    '''
    From data scraping to finish data clean
    '''
    
    ########## SCRAPING SECTION ##########
    # Get the dataframe from web scraping
    df = get_features(page, zipcodes)
    
    ########## CLEANING SECTION ##########
    df.reset_index(inplace=True)
    df.rename(columns={'index':'name'}, inplace=True)
    print('DATA FROM WEB SCRAPING:')
    df.info()

    # Drop duplicates and null values in name and price features
    df.drop_duplicates(inplace=True)
    df.dropna(subset = ['name', 'price'], inplace = True)

    # Drop abnormal mpgs
    df.drop(df[df.mpg > 150].index, inplace = True)

    # Fill null values with mean
    df.mpg.fillna((df.mpg).mean(), inplace = True)
    df.mileage.fillna((df.mileage).mean(), inplace = True)
    df.num_of_entertainment.fillna((df.num_of_entertainment).mean(), inplace = True)
    df.num_of_safety.fillna((df.num_of_safety).mean(), inplace = True)
  
    ##### Year, Make, and Model from Name #####
    # Create year and make columns
    df['year'] = df['name'].apply(lambda x: x[0:4])
    df['make'] = df['name'].apply(lambda x: x.split(' ')[1])

    # Complete truncated model name
    df.make = df.make.apply(complete_make)

    # Create model columns
    df['model'] = df['name'].apply(lambda x: x.split(' ')[2])

    ##### Drivtrain #####
    # Simplify drivetrain
    df.drivetrain = df.drivetrain.apply(simplify_drivetrain)

    # Drop unknown drivetrain
    df.drop(df[(df.drivetrain == '–') | (df.drivetrain == 'Unknown')].index, inplace=True)

    ##### Fuel Type #####
    # Simplify fuel type
    df.fuel_type = df.fuel_type.apply(simplify_fuel_type)

    # Drop unknown fuel type
    df.drop(df[df.fuel_type == '–'].index, inplace=True)

    ##### Transmission #####
    # Simplify transmission
    df.transmission = df.transmission.apply(simplify_transmission)

    ##### Engline #####
    df.engine = df.engine.apply(lambda x: x.strip('Engine: '))

    # Simplify engine
    df.engine = df.engine.apply(simplify_engine)

    ##### Convert float to int #####
    df.price = df.price.astype(int)
    df.mpg = df.mpg.astype(int)
    df.mileage = df.mileage.astype(int)
    df.num_of_entertainment = df.num_of_entertainment.astype(int)
    df.num_of_safety = df.num_of_safety.astype(int)

    ##### Check for outliers #####
    ### Price ###
    # Remove outliers
    df = df[df.price < 110000]
    
    # Price distribution after taking out outliers
    print('\n')
    sns.displot(df.price)
    plt.title('Price Distribution After Outlier Removal')

    ### Mileage ###
    # Remove outliers
    df = df[df.mileage < 200000]
    
    # Mileage distribution after taking out outliers
    print('\n')
    sns.displot(df.mileage)
    plt.title('Mileage Distribution After Outlier Removal')

    ##### Reset index #####
    df.reset_index(drop = True, inplace = True)

    ##### Re-arrange dataframe #####
    df = df[['name', 'price', 'mpg', 'mileage', 'num_of_entertainment',
         'num_of_safety', 'year','drivetrain', 'fuel_type', 
         'transmission', 'engine', 'make', 'model']]

    # Safe as csv for now
    df.to_csv('cars_data.csv')
    
    ########## DATA STORAGE SECTION ##########
    # Create connection
    engine = create_engine('sqlite:///test.db')
    
    # Bring up new dataframe
    df = pd.read_csv('cars_data.csv', index_col=0)
    print('\n')
    print('CLEANED DATA BEFORE SAVING IN SQL DATABASE')
    df.info()
    
    # Update the database with the new table
    df.to_sql(name='test', con=engine, if_exists='replace', index=False)
    
    # Bring up the new table
    df = pd.read_sql('SELECT * FROM test;', engine)
    print('\n')
    print('CLEANED DATA IN SQL DATABASE:')
    df.info()
    
    ########## FEATURE ENGINEERING ##########
    # Create subset with only numerical value
    df_num = df.iloc[:, 1:7]
    
    # Target/numerical pairplot
    sns.pairplot(df_num)
    plt.title('Target - Numerical Features Pairplot')
    
    # Correlation plot
    plt.figure(figsize = [15, 10])
    upper = np.triu(df_num.corr())
    sns.heatmap(df_num.corr(), cmap="YlGnBu", annot=True, vmin=-1, vmax=1, mask=upper)
    plt.xticks(rotation = 45)
    

