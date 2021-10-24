## Project MVP

The purpose of this project is to build an end to end used car sale price prediction application and help used car buyers/sellers to determine how much their cars worth based on the cars' features.

At this point, the model's hyperparameters are tuned, and the tuned model provided the following scores:

* Train R2 score: 0.99
* Test R2 score: 0.94
* MAE: 2658.98

A module called compy.py was created.  It takes in number of page and zip code as parameters, and then it performs web scraping, data cleaning, and database updating.  Because it took a long time to scrape the same amount of data for the original dataset (78,868 observations), the function only has been tested with small number of pages (four zip codes each with one page of ten car urls), and it worked properly.  

Two functions were created for the model training.  The first function allows hyperparameters in gridsearchCV be adjusted, and the second function performs the training with the best hyperparameters from the first function and provides the the test R2 score and MAE.

My next step is to create a web app with Streamlit.  And if time permits, test compy.py and the functions for model training with a bigger number of pages.
