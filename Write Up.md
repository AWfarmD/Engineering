## Predicting the Used Car Pricing

**Abstract**

The goal of this project is to build an end-to-end used car sales price predictor that can help used car buyers/sellers have a better idea on how much their cars worth before entering into the market.  I worked with data scraped from Cars.com, SQL database for data storage, Random Forest Regressor for model building, and Streamlit for web application to achieve this goal.

**Design**

Buying and selling an used car can be tricky for many people, especially when it comes to the price of the car.  Having a predicted price from a predictor, they can have a better idea on how much their cars worth before entering the market, and potentially have better negotiating power.

**Data**

The data of 89,717 listings were scraped from Cars.com.  After filling in missing numerical features with mean, dropping missing categorical features, and droping outliers for price and mileage, 75,253 listings with 5 numerical features and 6 categorical features were used for model building.

**Algorithms**

Random Forest Regressor was used to build a baseline model with train R2 score of 0.989 and validation R2 score of 0.924, and then the model's hyperparameters were tuned to provide a train, validation, and final test R2 scores of 0.991, 0.932, and 0.939 respectively.  The MAE for the final model is 2671.65.

The entire project was then connected with two modules to build a hyperparameter tuned model that is ready for web application.  With the first module that takes in the number of page of each city to be scraped and the city zip code as parameters, it is able to perfrom web scraping, data cleaning, database updating, and base model training.  

The output of the first module contains print out of dataframe information after web scraping, before going into SQL database, and after storing in database, so we can see how many data we drop during data clean and lose during database update.  It also prints out the distribution plots for price and mileage after removing outliers, so we can change the criteria for outlier removal if the plots doesn't not make sense.  Moverover, it prints out pair plot of price and all the features and correlation plots to help us with feature engineering if any is needed.  And last, it will print out the base model's train and validation R2 scores.

Then, the second modulel takes in lists of hyperparameters we'd like to tune and provides the best estimator, and then performs the training with the best estimator and provides the train, validation, and test R2 scores and MAE.

The final model was then used in Streamlit to create a web application which provides a predicted sale price for the car based on user's input of the car's features

**Tools**

* BeautifulSoup and Google Colab for web scraping
* Pandas for data cleaning and EDA
* Matplotlib and Seaborn for visualization
* SQL for data storage
* Sklearn for model building
* Streamlit for web application

**Communication**

Project pipeline and slides will be presented.