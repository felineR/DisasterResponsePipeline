# Disaster Response Pipeline

This project is created as part of Udacity's nanodegree program "Data Scientist". It is the second project of this program and is named "Disaster Response Pipelines".

The three main components of this project are:
1. ETL Pipeline | Extracts raw data from csv files, cleans, transforms and merges the raw data, and saves the preprocessed data in a database
2. Machine Learning Pipeline | Retrieves preprocessed data from database, splits it into training and test data, trains a machine learning model using natural language processing, 
				evaluates the model's performance, and saves the model as pickel file
3. Flask Web App | Runs a web application that shows message distributions and allows the user to enter a message that is categorized on 36 disaster related categories

Files in the repository:
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app
- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- DisasterResponse.db   # database to save clean data to
- models
|- train_classifier.py
|- classifier.pkl  # saved model 
- README.md

How to run the ETL pipeline:
- Open a new terminal window. Use terminal commands to navigate inside the data folder containing the process_data.py file.
- Type in the command line: python process_data.py disaster_messages.csv disaster_categories.csv

How to run the Machine Learning pipeline:
- Open a new terminal window. Use terminal commands to navigate inside the models folder containing the train_classifier.py file.
- Type in the command line: python train_classifier.py ../data/DisasterResponse.db classifier.pkl

How to run the web application:
- Open a new terminal window. Use terminal commands to navigate inside the app folder containing the run.py file.
- Type in the command line: python run.py
- Open another Terminal Window. Type: env|grep WORK
- In a new web browser window, type in the following: https://SPACEID-3001.SPACEDOMAIN 
   where SPACEID and SPACEDOMAIN are the variables retrieved from the second terminal window.

Python libraries used:
- sys
- pandas
- sqlite3
- sqlalchemy
- re
- nltk
- sklearn
- pickle
- json
- plotly
- flask

Acknowledgements:
- This project is conducted as part of the Udacity Nanodegree on Data Science. Please find further informations here: https://www.udacity.com/course/data-scientist-nanodegree--nd025

Author of this project is: F B
Created in January 2022