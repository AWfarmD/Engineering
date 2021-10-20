#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import pickle

from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV


# In[ ]:


def get_best_param(n_estimators, max_depth, min_samples_split, max_features):
    '''
    Get best hyperparameters for the model
    
    ***Each parameter is a list of values***
    '''
    
    # Load data
    engine = create_engine('sqlite:///test.db')
    df = pd.read_sql('SELECT * FROM test;', engine)
    
    # Get data ready
    df = pd.get_dummies(df, columns=['drivetrain', 'fuel_type', 
                                     'transmission','engine', 'make', 
                                     'model'], drop_first=True)
    
    X = df.drop(['name', 'price'], axis=1)
    y = df.price
    
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.25, random_state=42)
    
    # Get best param
    rf = RandomForestRegressor()

    param_grid = dict(n_estimators=n_estimators, max_depth=max_depth, 
                  min_samples_split=min_samples_split, 
                  max_features=max_features)

    grid_rf = GridSearchCV(rf, param_grid, cv=5, scoring='r2')
    grid_rf.fit(X_train, y_train)
    
    print('Best params: ', grid_rf.best_params_)
    print('Best estimator: ', grid_rf.best_estimator_)
    print('Best score: ', grid_rf.best_score_)


# In[ ]:


def scores_with_best_param(n_estimators, max_depth, min_samples_split, max_features):
    '''
    Get train R2 and validation R2 of the model with best hyperparameters
    
    ***Each parameter is an individual value***
    '''
    
    # Get data ready
    engine = create_engine('sqlite:///test.db')
    df = pd.read_sql('SELECT * FROM test;', engine)
    
    df = pd.get_dummies(df, columns=['drivetrain', 'fuel_type', 
                                     'transmission','engine', 'make', 
                                     'model'], drop_first=True)
    
    X = df.drop(['name', 'price'], axis=1)
    y = df.price
    
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model with combined train/validation data and best param
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                               min_samples_split=min_samples_split, 
                               max_features=max_features)
    
    rf.fit(X, y)
    
    # Safe the model
    with open('rf_final_model', 'wb') as to_write:
        pickle.dump(rf, to_write)
    
    # Scores
    test_score = rf.score(X_test, y_test)
    MAE = np.mean(np.abs(y_test - rf.predict(X_test)))
    
    print('Metrics of Final Model With Train/Validation Data Combined:')
    print(f'Test R^2: {test_score:.3f}')
    print(f'MAE: {MAE:.3f}')

