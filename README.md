# Python-EDA-ML-Predict-flight-price

Project Summary: Predictive Modeling for Flight Prices

This project focuses on predicting flight prices using machine learning techniques, specifically employing a RandomForestRegressor model. Here's a concise summary of the project's key steps and findings:

Project Steps:

Data Import and Exploration:

Imported a cleaned dataset using pandas and explored its structure, columns, data types, and basic statistics.
Investigated unique values of specific columns like airline, source city, destination city, departure time, arrival time, stops, and class.
Data Preprocessing:

Applied preprocessing steps including one-hot encoding for categorical variables (airline, source city, destination city, departure time).
Converted the 'class' column to binary (0 for Economy, 1 for other classes).
Converted 'stops' column values to numerical (0, 1, 2).
Dropped unnecessary columns ('Unnamed: 0', 'flight') and retained relevant columns ('duration', 'days_left', 'price').
Regression Model Training:

Utilized RandomForestRegressor for predicting flight prices.
Implemented K-Fold cross-validation (k=5) for model evaluation and prediction.
Evaluated model performance using metrics such as R² score, Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
Analysis and Visualization:

Analyzed MAE relative to the range of flight prices to gauge prediction accuracy.
Conducted residual analysis to understand model performance and consistency.
Assessed MAE against an acceptable range based on average flight price.
Explored feature importance to identify key factors influencing flight prices.
Prediction for New Data:

Developed a function to preprocess new data for prediction using the trained model.
Demonstrated predicting flight prices for new data based on the preprocessed features.
