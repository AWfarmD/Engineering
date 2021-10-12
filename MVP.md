## Engineering Project MVP

The purpose of this project is to help used car buyers/sellers to determine how much their cars worth based on the cars' features.

At this point, the model's hyperparameters are tuned, and the tuned model provides the following scores:

* Train R2 score: 0.992
* Test R2 score: 0.939
* MAE: 2671.649

A function called pipeline was created.  It takes in number of page and zip code as parameters, and then it performs web scraping, data cleaning, database updating, and base model training.  Because it takes a long time to scrape the same amount of data for the original dataset (78,868 observations), the function only has been tested with small set of data (40 observations from four zip codes each with one page of ten car urls), and it worked properly.  

Two functions are created for the hyperparameter tuning.  The first function allows hyperparameters in gridsearchCV be adjusted based on the test and validation performance of the base model (if the base model is overfitting or underfitting).  The second function performs the training with the best hyperparameters from the first function and provides the train and validation R2 scores. It also retrains the best estimator with the test and validation sets combined, and gives the test R2 score and MAE.

My next step is to create a web app with Streamlit.  And if time permits, test pipeline and the functions for hyperparameter tuning with a bigger set of data.
