# Project Write-Up

### Abstract

The goal of this project is to build an end-to-end used car sales price prediction web application that can help used car buyers/sellers have a better idea on how much their cars worth before entering into the market.  I worked with data scraped from Cars.com, SQLite for data storage, Random Forest Regressor for model building, and Streamlit for web application to achieve this goal.

### Design

Buying and selling an used car can be tricky for many people, especially when it comes to the price of the car.  It is hard to know if the value of the car really worth the price on the window sticker.  Having a predicted price from a predictor, they can have a better idea on how much their cars are worth before entering into the market, and potentially have better negotiating power.

### Data

The data of 78,868 listings were scraped from Cars.com.  After data cleaning and wrangling, 75,253 listings were used for model building.  With price as target, the data has twelve features: miles per gallon, mileage of the car, number of entertainment features, number of safety features, model year, type of drivetrain, fuel type, type of transmission, type of engine, make of the car, and model of the car.  

### Algorithms

**Cleaning & EDA**

At this stage, I engineered some features.  Model year, make of the car, and model of the car were extracted from the name of the car information page.  Filled NaN values in miles per gallon, mileage of the car, number of entertainment features, and number of safety features with the mean of each feature.  I also dropped any listings that have NaN values for the name of the car informaton page (where I wasn't able to get model year, make of the car, and model of the car) and for the price from the dataset.  Lastly, I dropped outliers that have the price of the car greater than $110,000 dollars and the mileage of the car greater than 200,000 miles.  This left 75,253 listings in the dataset.

**Model Training**

Random Forest Regressor was chosen to build a model because it works well with non-numerical data and has relative fewer hyperparameters to tune, so it can yield a model with great performance and takes less time to train when the data are updated.  The entire dataset consisted of numerical and dummified categorical features were used for model training, and the model was trained on the training data (comprised of 80% of the full dataset) with the best values for number of estimators, max depth, minimum sample split, and max features from GridSearchCV.  The resulting model attained R2 score of 0.94 and a MAE of 2658.98.

**Pipeline**

The entire project was then connected with two modules to build a hyperparameter tuned model that is ready for web application.  With the first module (compy.py) that takes in the city zip code and the number of page of each city to be scraped as parameters, it is able to perfrom web scraping, data cleaning, and database updating.  

The output of the compy module contains print out of dataframe information after web scraping, before going into SQL database, and after storing in database, so we can see how many data we lose during data cleaning and database update, and take action when necessary.  It also prints out the distribution plots for price and mileage after removing outliers, so we can make sure the outliers are removed correctly.  At last, it prints out pair plot of price and all the features and correlation plots to help us with feature engineering if any is needed with the updated data.  

The second module (tunepy.py) takes in lists of hyperparameters (number of estimators, max depth, minimum sample split, and max features) we would like to tune and provides the best estimator, and then performs the training with the best estimator and outputs the R2 score and MAE on test data.

The final model was then used in Streamlit to create a web application which provides a predicted sale price for the car based on user's input of the car's features

### Tools

* BeautifulSoup and Google Colab for web scraping
* Pandas for data cleaning and EDA
* Matplotlib and Seaborn for visualization
* SQLite for data storage
* Sklearn for model building
* Streamlit for web application

### Communication

Project pipeline, visualization, and slides will be presented.